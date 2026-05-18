"""
tree_search.py — Reasoning-based retrieval over a PageIndex tree.

The retrieval loop:
  1. Read the tree index (in-context).
  2. Reason over it: which node_ids are most likely to contain the answer?
  3. Fetch raw text for those nodes.
  4. Check: is the information sufficient?
     └─ Yes → generate final answer.
     └─ No  → loop back with updated context (up to max_iterations).
  5. Return the answer + metadata.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .config import PageIndexConfig
from .groq_client import chat, chat_json
from .prompts import (
    CHECK_SUFFICIENT,
    FINAL_ANSWER,
    SELECT_NODES,
    SYS_RETRIEVER,
)
from .utils.logging import trail

log = logging.getLogger(__name__)


# ─── Result type ──────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """
    Returned by TreeSearcher.answer().

    Attributes
    ----------
    text              : The final answer string.
    retrieved_nodes   : List of node dicts that were read.
    reasoning_steps   : Step-by-step reasoning trace.
    iterations        : How many retrieval loops were needed.
    """
    text:             str
    retrieved_nodes:  List[Dict[str, Any]] = field(default_factory=list)
    reasoning_steps:  List[str]            = field(default_factory=list)
    iterations:       int                  = 0

    def __str__(self):
        return self.text


# ─── TreeSearcher ─────────────────────────────────────────────────────────

# Trees with this many leaf nodes or fewer get ALL leaves read
# without iterative LLM selection — eliminates missed siblings.
SMALL_TREE_THRESHOLD = 6

# Phrases in LLM reasoning that signal a document is irrelevant
_IRRELEVANCE_SIGNALS = (
    "no relevant",
    "not relevant",
    "no direct",
    "not directly relevant",
    "no nodes",
    "none of the nodes",
    "no match",
    "unrelated",
    "not applicable",
    "no information",
)


class TreeSearcher:
    """
    Performs reasoning-based document retrieval over a PageIndex tree.

    Parameters
    ----------
    tree   : The dict returned by build_tree()
    config : PageIndexConfig (defaults used if None)

    The `pages` argument is optional — if you built the tree with
    add_node_text=True, the raw text is already embedded in the tree
    and pages are not needed.
    """

    def __init__(
        self,
        tree: Dict[str, Any],
        config: Optional[PageIndexConfig] = None,
        pages: Optional[List[str]] = None,
    ):
        self.tree   = tree
        self.config = config or PageIndexConfig()
        self.pages  = pages or []

        # Build flat id→node index for O(1) lookup
        self._index: Dict[str, Dict] = {}
        self._build_index(tree.get("nodes", []))

        # Pre-compute leaf nodes (nodes with no children)
        self._leaf_ids: List[str] = [
            nid for nid, node in self._index.items()
            if not node.get("nodes")
        ]

        # Keyword agent for large-tree pre-filtering.
        # Lazy import avoids any potential circular dependency since
        # agents/hunter.py imports TreeSearcher but keyword_agent.py does not.
        from .agents.keyword_agent import KeywordAgent
        self._keyword_agent = KeywordAgent(
            model=self.config.model,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            quiet=self.config.quiet,
            enable_thinking=self.config.enable_thinking,
            num_ctx=self.config.num_ctx,
        )

    # ── Public ───────────────────────────────────────────────────────────

    def answer(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        pre_visited: Optional[set] = None,
    ) -> SearchResult:
        """
        Answer `question` using agentic tree search.

        Parameters
        ----------
        question    : The user's question (free text)
        history     : Prior conversation turns as a list of
                      {"role": "user"/"assistant", "content": "..."} dicts.
                      Used for context-aware multi-turn retrieval.
        pre_visited : Node IDs to treat as already visited (skip them).
                      Used by the iterative retrieval loop to avoid
                      re-fetching the same nodes in subsequent rounds.

        Returns
        -------
        SearchResult  (str-able: just do str(result) or result.text)
        """
        history = history or []
        gathered:  List[Tuple[str, str]] = []   # (node_id, text)
        visited:   set                   = set(pre_visited or set())
        steps:     List[str]             = []
        nodes_out: List[Dict]            = []

        # ── Small-tree fast-path ──────────────────────────────────────
        # If the tree has few leaf nodes, read ALL of them at once.
        # This eliminates the risk of missing important sibling nodes
        # (e.g. doc4/0002 "Operational AI Systems" being skipped when
        # the LLM picks its sibling 0003 "Autonomous Workflows" first).
        unvisited_leaves = [lid for lid in self._leaf_ids if lid not in visited]
        if len(unvisited_leaves) <= SMALL_TREE_THRESHOLD:
            log.debug(f"Small tree ({len(unvisited_leaves)} unvisited leaves) — reading all")
            steps.append(
                f"[fast-path] Small tree with {len(unvisited_leaves)} unvisited "
                f"leaf nodes — reading all to maximise recall."
            )
            for nid in unvisited_leaves:
                visited.add(nid)
                node = self._index.get(nid)
                if node is None:
                    continue
                text = self._get_text(node)
                if text and text.strip():
                    gathered.append((nid, text))
                    nodes_out.append(node)

            # Skip to answer generation — no iterative loop needed
            context = _join_gathered(gathered, with_titles=True, index=self._index)
            answer  = self._answer(question, context)
            return SearchResult(
                text=answer,
                retrieved_nodes=nodes_out,
                reasoning_steps=steps,
                iterations=1,
            )

        # ── Standard iterative retrieval ──────────────────────────────
        # For large trees, run keyword expansion + local search first to
        # build a compact candidate sub-tree (saves ~95% of prompt tokens).
        pre_filter_enabled   = getattr(self.config, 'enable_pre_filtering', True)
        pre_filter_threshold = getattr(self.config, 'pre_filter_threshold', 50)
        max_candidates       = getattr(self.config, 'max_filter_candidates', 20)
        use_pre_filter       = (
            pre_filter_enabled and len(self._index) > pre_filter_threshold
        )

        if use_pre_filter:
            trail.step(
                "PRE-FILTER INITIATED",
                f"Tree size ({len(self._index)} nodes) exceeds threshold ({pre_filter_threshold}).\n"
                f"Running keyword pre-filtering to build a compact candidate sub-tree.",
                quiet=self.config.quiet
            )
            log.info(
                f"[pre-filter] ACTIVATED — {len(self._index)} nodes "
                f"(threshold={pre_filter_threshold}). Calling KeywordAgent ..."
            )
            steps.append(
                f"[pre-filter] Tree has {len(self._index)} nodes — "
                f"expanding keywords for local candidate search ..."
            )

            doc_context = self.tree.get("document_description", "")
            expanded_kws = self._keyword_agent.expand(
                question, history, doc_context=doc_context
            )
            matched_ids = self._local_node_search(expanded_kws)

            log.info(
                f"[pre-filter] KeywordAgent produced {len(expanded_kws)} terms → "
                f"{len(matched_ids)} matching nodes. "
                f"Terms: {expanded_kws[:8]}"
            )
            steps.append(
                f"[pre-filter] {len(expanded_kws)} terms → "
                f"{len(matched_ids)} matching nodes"
            )

            if matched_ids:
                candidate_ids     = matched_ids[:max_candidates]
                candidate_subtree = self._build_candidate_subtree(candidate_ids)
                log.info(
                    f"[pre-filter] COMPLETE — built candidate sub-tree with "
                    f"{len(candidate_ids)} seed nodes + ancestors. "
                    f"Top seeds: {candidate_ids[:5]}"
                )
                steps.append(
                    f"[pre-filter] Candidate sub-tree: "
                    f"top-{len(candidate_ids)} nodes + ancestors"
                )
                trail.step(
                    "PRE-FILTER COMPLETED",
                    f"Filtered {len(self._index)} nodes → {len(candidate_ids)} "
                    f"candidate seed nodes + their ancestors.",
                    {
                        "total_tree_nodes": len(self._index),
                        "candidate_seed_ids": candidate_ids,
                        "matched_keywords": expanded_kws,
                    },
                    quiet=self.config.quiet
                )
                tree_json = json.dumps(candidate_subtree, indent=2)
            else:
                # No keyword hits anywhere — fall back to full compact tree
                log.warning(
                    f"[pre-filter] ZERO matches across {len(self._index)} nodes "
                    f"for terms: {expanded_kws[:10]}. "
                    f"Falling back to full compact tree — check trail.log for details."
                )
                steps.append("[pre-filter] No matches — falling back to full tree")
                trail.step(
                    "PRE-FILTER COMPLETED (NO MATCHES)",
                    f"Zero keyword matches found across {len(self._index)} nodes. "
                    f"Falling back to the full compact tree.",
                    {"searched_terms": expanded_kws},
                    quiet=self.config.quiet
                )
                tree_json = json.dumps(self._compact_tree(), indent=2)
        else:
            log.debug(
                f"[pre-filter] SKIPPED — {len(self._index)} nodes "
                f"(<= threshold {pre_filter_threshold}) or pre-filtering disabled."
            )
            tree_json = json.dumps(self._compact_tree(), indent=2)

        max_iter = self.config.max_retrieval_iterations

        for i in range(1, max_iter + 1):
            # 1. Select nodes
            sel     = self._select(question, tree_json, history, gathered, visited)
            reason  = sel.get("reasoning", "")
            ids     = [nid for nid in sel.get("node_ids", []) if nid not in visited]

            steps.append(f"[{i}] {reason}")
            log.debug(f"Iter {i}: selected {ids}")

            if not ids:
                # ── Relevance-gated early exit ────────────────────────
                # If the very first iteration returns no IDs AND the
                # reasoning signals irrelevance, bail out immediately.
                if i == 1 and any(sig in reason.lower() for sig in _IRRELEVANCE_SIGNALS):
                    log.debug(f"Document flagged as irrelevant: {reason}")
                    steps.append(f"[early-exit] Document irrelevant: {reason}")
                break

            # 2. Fetch text
            for nid in ids:
                visited.add(nid)
                node = self._index.get(nid)
                if node is None:
                    continue
                text = self._get_text(node)
                gathered.append((nid, text))
                nodes_out.append(node)

            # 3. Sufficiency check
            combined = _join_gathered(gathered)
            if self._sufficient(question, combined):
                break

        # 4. Final answer
        context = _join_gathered(gathered, with_titles=True, index=self._index)
        answer  = self._answer(question, context)

        return SearchResult(
            text=answer,
            retrieved_nodes=nodes_out,
            reasoning_steps=steps,
            iterations=i,
        )

    def get_node(self, node_id: str) -> Optional[Dict]:
        """Return a node dict by node_id, or None."""
        return self._index.get(node_id)

    def get_text(self, node_id: str) -> Optional[str]:
        """Return the raw text for a node_id, or None."""
        node = self._index.get(node_id)
        return self._get_text(node) if node else None

    def nodes(self) -> List[Dict]:
        """Flat list of all nodes (no children)."""
        return [
            {k: v for k, v in n.items() if k != "nodes"}
            for n in self._index.values()
        ]

    # ── Internal ─────────────────────────────────────────────────────────

    def _build_index(self, nodes: List[Dict]) -> None:
        for n in nodes:
            nid = n.get("node_id")
            if nid:
                self._index[nid] = n
            self._build_index(n.get("nodes", []))

    def _compact_tree(self) -> List[Dict]:
        """Tree without raw text fields — keeps the prompt small."""
        def _strip(nodes):
            out = []
            for n in nodes:
                item = {k: v for k, v in n.items() if k not in ("text", "nodes")}
                kids = _strip(n.get("nodes", []))
                if kids:
                    item["nodes"] = kids
                out.append(item)
            return out
        return _strip(self.tree.get("nodes", []))

    def _get_text(self, node: Dict) -> str:
        if node.get("text"):
            return node["text"]
        if self.pages:
            s = node.get("start_index", 0)
            e = node.get("end_index", s + 1)
            # Ensure at least one page is included even when end_index == start_index
            e = max(e, s + 1)
            return "\n\n".join(self.pages[s:e])
        return f"[Pages {node.get('start_index')}–{node.get('end_index')}]"

    # ── Candidate sub-tree pre-filtering ─────────────────────────────────

    def _local_node_search(self, keywords: List[str]) -> List[str]:
        """
        Score every indexed node by keyword hit-count using fast case-insensitive
        regex matching across title (weighted 3×), summary (2×), and a text
        preview (1×, first 600 chars).  Zero LLM calls — pure local search.

        Returns node IDs ranked best-match first.
        """
        if not keywords:
            return []

        # De-duplicate while preserving insertion order
        unique_kws = list(dict.fromkeys(keywords))
        patterns: List[re.Pattern] = []
        for kw in unique_kws:
            try:
                patterns.append(re.compile(re.escape(kw), re.IGNORECASE))
            except re.error:
                pass
        if not patterns:
            return []

        scored: List[Tuple[int, str]] = []
        for nid, node in self._index.items():
            title   = node.get("title", "")
            summary = node.get("summary", "")
            text    = node.get("text", "")[:600]
            # Title weighted 3×, summary 2×, text preview 1×
            searchable = f"{title} {title} {title} {summary} {summary} {text}"
            hits = sum(1 for pat in patterns if pat.search(searchable))
            if hits > 0:
                scored.append((hits, nid))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [nid for _, nid in scored]

    def _build_parent_map(
        self,
        nodes: List[Dict],
        parent_id: Optional[str] = None,
    ) -> Dict[str, Optional[str]]:
        """Recursively build a {node_id → parent_id} lookup for the entire tree."""
        result: Dict[str, Optional[str]] = {}
        for node in nodes:
            nid = node.get("node_id")
            if nid:
                result[nid] = parent_id
                result.update(
                    self._build_parent_map(node.get("nodes", []), nid)
                )
        return result

    def _build_candidate_subtree(self, matched_ids: List[str]) -> List[Dict]:
        """
        Given top-N matched node IDs, reconstruct a pruned, compact sub-tree
        containing those nodes AND every ancestor up to the document root.

        Ancestors are included so the LLM retains full hierarchical context
        (e.g. knowing node 0042 belongs to "Section 4: Financial Risks"
        rather than "Section 12: Appendix").

        Returns text-free nodes — identical format to _compact_tree() — so
        the result slots directly into the SELECT_NODES prompt.
        """
        if not matched_ids:
            return []

        parent_map = self._build_parent_map(self.tree.get("nodes", []))

        # Walk each matched node up to the root, collecting all ancestor IDs
        include_ids: set = set()
        for mid in matched_ids:
            current: Optional[str] = mid
            while current is not None:
                include_ids.add(current)
                current = parent_map.get(current)   # None signals root reached

        # Recursively filter the original tree, keeping only included nodes
        def _filter(nodes: List[Dict]) -> List[Dict]:
            out: List[Dict] = []
            for node in nodes:
                nid = node.get("node_id")
                if nid not in include_ids:
                    continue
                compact = {
                    k: v for k, v in node.items()
                    if k not in ("text", "nodes")
                }
                children = _filter(node.get("nodes", []))
                if children:
                    compact["nodes"] = children
                out.append(compact)
            return out

        return _filter(self.tree.get("nodes", []))

    def _select(
        self,
        question: str,
        tree_json: str,
        history: List[Dict],
        gathered: List[Tuple[str, str]],
        visited: set,
    ) -> Dict:
        history_block = (
            "Conversation history:\n" +
            "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in history[-6:])
            if history else ""
        )
        visited_block = (
            f"Already read nodes (skip these): {', '.join(visited)}"
            if visited else ""
        )
        prompt = SELECT_NODES.format(
            tree=tree_json,
            history_block=history_block,
            question=question,
            visited_block=visited_block,
        )
        try:
            return chat_json(
                prompt,
                model=self.config.model,
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                system=SYS_RETRIEVER,
                temperature=0.0,
                max_tokens=256,
                quiet=self.config.quiet,
                enable_thinking=self.config.enable_thinking,
                num_ctx=self.config.num_ctx,
            )
        except Exception as e:
            log.warning(f"Node selection failed: {e}")
            # Fallback: first unvisited node
            for nid in self._index:
                if nid not in visited:
                    return {"reasoning": "fallback", "node_ids": [nid]}
            return {"reasoning": "no nodes", "node_ids": []}

    def _sufficient(self, question: str, gathered: str) -> bool:
        prompt = CHECK_SUFFICIENT.format(
            question=question,
            gathered=gathered[:8_000],
        )
        try:
            r = chat_json(
                prompt,
                model=self.config.model,
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                system=SYS_RETRIEVER,
                temperature=0.0,
                max_tokens=64,
                quiet=self.config.quiet,
                enable_thinking=self.config.enable_thinking,
                num_ctx=self.config.num_ctx,
            )
            return bool(r.get("sufficient", False))
        except Exception:
            return True   # on parse error, don't loop forever

    def _answer(self, question: str, context: str) -> str:
        prompt = FINAL_ANSWER.format(
            question=question,
            context=context[:16_000],
        )
        try:
            return chat(
                prompt,
                model=self.config.model,
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                temperature=self.config.temperature,
                max_tokens=self.config.max_output_tokens,
                quiet=self.config.quiet,
                enable_thinking=self.config.enable_thinking,
                num_ctx=self.config.num_ctx,
            ).strip()
        except Exception as e:
            return f"[Error generating answer: {e}]"


# ─── Helpers ──────────────────────────────────────────────────────────────

def _dedup_text(text: str) -> str:
    """Remove repeated sentences."""
    if not text or len(text) < 100:
        return text
    # Fix missing whitespace after sentence-ending punctuation
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    seen = set()
    unique = []
    for s in sentences:
        key = re.sub(r'\s+', ' ', s.strip().lower())
        if not key:
            continue
        if key not in seen:
            seen.add(key)
            unique.append(s.strip())
    return ' '.join(unique)


def _join_gathered(
    gathered: List[Tuple[str, str]],
    with_titles: bool = False,
    index: Optional[Dict] = None,
) -> str:
    parts = []
    for nid, text in gathered:
        clean = _dedup_text(text)
        if with_titles and index:
            title = index.get(nid, {}).get("title", nid)
            parts.append(f"[Node {nid} — {title}]\n{clean}")
        else:
            parts.append(f"[Node {nid}]\n{clean}")
    return "\n\n---\n\n".join(parts)