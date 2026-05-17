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
        tree_json  = json.dumps(self._compact_tree(), indent=2)
        max_iter   = self.config.max_retrieval_iterations

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