"""
hunter.py — Hunter Agent.

The Hunter Agent is a wrapper around the existing TreeSearcher.
It searches a single document's tree index and returns the relevant
text chunks with their source metadata.

In the orchestrator, multiple Hunter Agents run concurrently
(one per document) using a thread pool for parallel retrieval.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..config import PageIndexConfig
from ..storage.base import TreeStore
from ..tree_search import TreeSearcher

log = logging.getLogger(__name__)


@dataclass
class HuntResult:
    """
    Result from searching a single document.

    Attributes
    ----------
    doc_id          : Which document was searched.
    doc_title       : Document title (for citations).
    chunks          : List of retrieved text chunks with node metadata.
    reasoning_steps : Step-by-step reasoning trace from the TreeSearcher.
    iterations      : How many retrieval loops were needed.
    success         : Whether the search completed without error.
    error           : Error message if failed.
    """
    doc_id:          str
    doc_title:       str             = ""
    chunks:          List[Dict]      = field(default_factory=list)
    reasoning_steps: List[str]       = field(default_factory=list)
    iterations:      int             = 0
    success:         bool            = True
    error:           str             = ""


class HunterAgent:
    """
    Searches individual document trees to find relevant text chunks.

    Parameters
    ----------
    store   : TreeStore where indexed trees are stored.
    config  : PageIndexConfig.
    """

    def __init__(
        self,
        store: TreeStore,
        config: PageIndexConfig,
    ):
        self.store  = store
        self.config = config

    def hunt(
        self,
        doc_id: str,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        exclude_nodes: Optional[set] = None,
        pre_expanded_keywords: Optional[List[str]] = None,
    ) -> HuntResult:
        """
        Search a single document's tree for the answer.

        Parameters
        ----------
        doc_id        : The document to search.
        question      : The user's question.
        history       : Optional conversation history.
        exclude_nodes : Node IDs to skip (already retrieved in prior rounds).

        Returns
        -------
        HuntResult with text chunks and metadata.

        Notes
        -----
        If the TreeSearcher's final answer begins with "NO_INFO", it means
        the retrieved nodes contained no relevant information for this question.
        In that case we discard ALL chunks for this document so the Orchestrator
        does not forward irrelevant text to the Synthesizer, preventing token
        bloat and rate-limit errors.
        """
        try:
            # Load the tree from storage
            tree = self.store.load(doc_id)
            doc_title = tree.get("document_description", doc_id)[:100]

            # Create a TreeSearcher for this document
            searcher = TreeSearcher(tree, config=self.config)
            result = searcher.answer(
                question,
                history=history,
                pre_visited=exclude_nodes,
                pre_expanded_keywords=pre_expanded_keywords,
            )

            # ── NO_INFO gate ──────────────────────────────────────────
            # The FINAL_ANSWER prompt instructs the LLM to begin its
            # response with exactly "NO_INFO" when none of the retrieved
            # sections contain relevant information.  Forwarding those
            # chunks to the Synthesizer would inflate the prompt with
            # useless text and can push the request over the TPM limit.
            # We drop them here so only signal-bearing chunks survive.
            if result.text.startswith("NO_INFO"):
                log.info(
                    f"[hunter] doc='{doc_id}' returned NO_INFO — "
                    f"discarding {len(result.retrieved_nodes)} node(s) "
                    f"as irrelevant to prevent Synthesizer token bloat."
                )
                return HuntResult(
                    doc_id=doc_id,
                    doc_title=doc_title,
                    chunks=[],          # no chunks — document is irrelevant
                    reasoning_steps=result.reasoning_steps
                    + [f"[NO_INFO] Document '{doc_id}' pruned — no relevant content found."],
                    iterations=result.iterations,
                    success=True,       # not an error; document was searched successfully
                )

            # Package the chunks with source metadata
            chunks = []
            for node in result.retrieved_nodes:
                text = searcher.get_text(node.get("node_id", ""))
                chunks.append({
                    "doc_id": doc_id,
                    "doc_title": doc_title,
                    "node_id": node.get("node_id", ""),
                    "node_title": node.get("title", ""),
                    "start_page": node.get("start_index", 0),
                    "end_page": node.get("end_index", 0),
                    "text": text or "",
                })

            # Deduplicate chunks with identical text (e.g. child nodes
            # sharing the same page produce identical content).  Merge
            # their titles so the synthesizer sees the content once.
            chunks = self._dedup_chunks(chunks)

            return HuntResult(
                doc_id=doc_id,
                doc_title=doc_title,
                chunks=chunks,
                reasoning_steps=result.reasoning_steps,
                iterations=result.iterations,
                success=True,
            )

        except FileNotFoundError:
            return HuntResult(
                doc_id=doc_id,
                success=False,
                error=f"Tree not found for doc_id='{doc_id}'",
            )
        except Exception as e:
            log.error(f"Hunt failed for {doc_id}: {e}")
            return HuntResult(
                doc_id=doc_id,
                success=False,
                error=str(e),
            )

    @staticmethod
    def _dedup_chunks(chunks: List[Dict]) -> List[Dict]:
        """Merge chunks with identical text into one, combining titles."""
        seen: Dict[str, Dict] = {}  # text hash -> merged chunk
        order: List[str] = []       # preserve insertion order

        for chunk in chunks:
            text = chunk.get("text", "")
            key = text[:200]  # use first 200 chars as dedup key
            if key in seen:
                # Merge title
                existing = seen[key]
                existing_title = existing.get("node_title", "")
                new_title = chunk.get("node_title", "")
                if new_title and new_title not in existing_title:
                    existing["node_title"] = f"{existing_title} | {new_title}"
                # Expand page range
                existing["start_page"] = min(
                    existing.get("start_page", 0),
                    chunk.get("start_page", 0),
                )
                existing["end_page"] = max(
                    existing.get("end_page", 0),
                    chunk.get("end_page", 0),
                )
            else:
                seen[key] = dict(chunk)  # copy
                order.append(key)

        return [seen[k] for k in order]

    def hunt_parallel(
        self,
        doc_ids: List[str],
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        max_workers: int = 5,
        exclude_nodes: Optional[set] = None,
        parallel: bool = True,
        pre_expanded_keywords: Optional[List[str]] = None,
    ) -> List[HuntResult]:
        """
        Search multiple documents in parallel using a thread pool.

        Parameters
        ----------
        doc_ids       : List of document IDs to search.
        question      : The user's question.
        history       : Optional conversation history.
        max_workers   : Max concurrent threads.
        exclude_nodes : Node IDs to skip (already retrieved in prior rounds).
        parallel      : If False, hunt sequentially instead of using threads.

        Returns
        -------
        List of HuntResult, one per document.
        """
        results: List[HuntResult] = []

        if not parallel:
            for doc_id in doc_ids:
                try:
                    result = self.hunt(
                        doc_id, question, history,
                        exclude_nodes, pre_expanded_keywords,
                    )
                    results.append(result)
                except Exception as e:
                    results.append(HuntResult(
                        doc_id=doc_id,
                        success=False,
                        error=str(e),
                    ))
            return results

        with ThreadPoolExecutor(max_workers=min(max_workers, len(doc_ids))) as pool:
            futures = {
                pool.submit(
                    self.hunt, doc_id, question, history,
                    exclude_nodes, pre_expanded_keywords,
                ): doc_id
                for doc_id in doc_ids
            }
            for future in as_completed(futures):
                doc_id = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append(HuntResult(
                        doc_id=doc_id,
                        success=False,
                        error=str(e),
                    ))

        return results

    def quick_score(self, doc_id: str, keywords: List[str]) -> int:
        """
        Return the max keyword score for a tree without any LLM calls.
        Used by the Orchestrator to rank and prune candidate trees before
        launching expensive parallel hunters.

        Parameters
        ----------
        doc_id   : Document to score.
        keywords : Pre-expanded keyword list from the KeywordAgent.

        Returns
        -------
        Integer score (0 = no signal, higher = stronger match).
        """
        try:
            tree = self.store.load(doc_id)
            searcher = TreeSearcher(tree, config=self.config)
            return searcher.max_keyword_score(keywords)
        except Exception:
            return 0

    def quick_score_all(
        self,
        doc_ids: List[str],
        keywords: List[str],
        max_workers: int = 8,
    ) -> Dict[str, int]:
        """
        Score all candidate trees in parallel (pure local regex, no LLM).

        Returns
        -------
        Dict mapping doc_id → score.  Missing entries default to 0.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        scores: Dict[str, int] = {}
        if not doc_ids or not keywords:
            return {did: 0 for did in doc_ids}

        with ThreadPoolExecutor(max_workers=min(max_workers, len(doc_ids))) as pool:
            futures = {pool.submit(self.quick_score, did, keywords): did for did in doc_ids}
            for future in as_completed(futures):
                did = futures[future]
                try:
                    scores[did] = future.result()
                except Exception:
                    scores[did] = 0

        return scores

    def hunt_parallel_with_excludes(
        self,
        doc_ids: List[str],
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        max_workers: int = 5,
        exclude_fn: Optional[Any] = None,
        parallel: bool = True,
    ) -> List[HuntResult]:
        """
        Search multiple documents in parallel, with per-document exclude sets.

        This solves a critical bug where node_ids are only unique within a
        document (e.g. doc1 and doc4 both have node '0002').  A shared
        exclude_nodes set would silently drop chunks from later documents.

        Parameters
        ----------
        doc_ids    : List of document IDs to search.
        question   : The user's question.
        history    : Optional conversation history.
        max_workers: Max concurrent threads.
        exclude_fn : Callable(doc_id) -> set of node_ids to exclude for
                     that specific document.  If None, no nodes are excluded.
        parallel   : If False, hunt sequentially instead of using threads.

        Returns
        -------
        List of HuntResult, one per document.
        """
        results: List[HuntResult] = []

        if not parallel:
            for doc_id in doc_ids:
                try:
                    per_doc_exclude = exclude_fn(doc_id) if exclude_fn else None
                    result = self.hunt(doc_id, question, history, per_doc_exclude)
                    results.append(result)
                except Exception as e:
                    results.append(HuntResult(
                        doc_id=doc_id,
                        success=False,
                        error=str(e),
                    ))
            return results

        with ThreadPoolExecutor(max_workers=min(max_workers, len(doc_ids))) as pool:
            futures = {}
            for doc_id in doc_ids:
                per_doc_exclude = exclude_fn(doc_id) if exclude_fn else None
                future = pool.submit(
                    self.hunt, doc_id, question, history, per_doc_exclude
                )
                futures[future] = doc_id

            for future in as_completed(futures):
                doc_id = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append(HuntResult(
                        doc_id=doc_id,
                        success=False,
                        error=str(e),
                    ))

        return results

