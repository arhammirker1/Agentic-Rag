"""
planner.py — Planner Agent.

The Planner is the first agent in the loop.  It receives the user's question,
queries the document graph to find relevant documents, and returns a ranked
list of doc_ids for the Hunter agents to search.

Flow:
  1. Use the LLM to extract key search terms from the question.
  2. Query the graph (by topics + full-text) to find candidate documents.
  3. Use the LLM to rank/filter candidates based on their summaries.
  4. Return the top-N doc_ids.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..graph.base import DocNode, DocumentGraph
from ..groq_client import chat_json

log = logging.getLogger(__name__)


# ── Prompts ───────────────────────────────────────────────────────────────

EXTRACT_SEARCH_TERMS = """\
You are a search query planner.  Given a user's question, extract the
key search terms and topics that would help find relevant documents.

Question: "{question}"

{history_block}

Return JSON:
{{
  "search_terms": ["<term1>", "<term2>", ...],
  "topics": ["<broad topic1>", "<broad topic2>", ...]
}}

Rules:
- "search_terms" are specific keywords or phrases to match.
- "topics" are broader themes/categories.
- Include 2-8 terms in each list.
"""

RANK_DOCUMENTS = """\
You are selecting the most relevant documents to answer a question.

Question: "{question}"

Available documents:
{doc_summaries}

Select the documents most likely to contain the answer.
Return JSON:
{{
  "reasoning": "<one sentence explaining your selection>",
  "doc_ids": ["{example_id}", ...]
}}

Rules:
- Select 1 to {max_docs} documents.
- Prefer documents whose summary directly relates to the question.
- If no document seems relevant, return an empty list.
"""


@dataclass
class PlanResult:
    """Output of the Planner Agent."""
    doc_ids:   List[str]       = field(default_factory=list)
    reasoning: str             = ""
    doc_nodes: List[DocNode]   = field(default_factory=list)


class PlannerAgent:
    """
    Selects which documents to search based on graph metadata.

    Parameters
    ----------
    graph    : The document graph backend.
    model    : Groq model ID.
    api_key  : Groq API key.
    max_docs : Maximum documents to select per query.
    """

    def __init__(
        self,
        graph: DocumentGraph,
        *,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_docs: int = 5,
        quiet: bool = False,
        enable_thinking: bool = False,
        num_ctx: int = 32768,
    ):
        self.graph    = graph
        self.model    = model
        self.api_key  = api_key
        self.base_url = base_url
        self.max_docs = max_docs
        self.quiet    = quiet
        self.enable_thinking = enable_thinking
        self.num_ctx  = num_ctx

    def plan(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> PlanResult:
        """
        Given a question, select the most relevant documents from the graph.

        Parameters
        ----------
        question : The user's question.
        history  : Optional conversation history for context.

        Returns
        -------
        PlanResult with doc_ids, reasoning, and doc_nodes.
        """
        history = history or []

        # If graph is small (≤ max_docs), skip planning — search all docs
        total = self.graph.count()
        if total == 0:
            return PlanResult(reasoning="No documents in the index.")

        if total <= self.max_docs:
            all_docs = self.graph.list_documents()
            # Filter out parent-only grouping nodes (they have no tree
            # stored — only their sub-tree children do).  A parent node
            # is one that has children with parent_doc_id pointing to it.
            parent_ids = {d.parent_doc_id for d in all_docs if d.parent_doc_id}
            searchable = [d for d in all_docs if d.doc_id not in parent_ids]
            return PlanResult(
                doc_ids=[d.doc_id for d in searchable],
                reasoning=f"Searching all {len(searchable)} documents.",
                doc_nodes=searchable,
            )

        # 1. Extract search terms from the question
        terms = self._extract_terms(question, history)
        search_terms = terms.get("search_terms", [])
        topics = terms.get("topics", [])

        # 2. Query the graph
        candidates: Dict[str, DocNode] = {}

        # Search by topics (wider limit to catch sub-tree parts)
        for doc in self.graph.search_by_topics(topics, limit=self.max_docs * 4):
            candidates[doc.doc_id] = doc

        # Search by text
        for term in search_terms[:5]:
            for doc in self.graph.search_by_text(term, limit=self.max_docs * 2):
                candidates[doc.doc_id] = doc

        if not candidates:
            # Fallback: return all documents up to max_docs
            all_docs = self.graph.list_documents()[:self.max_docs]
            return PlanResult(
                doc_ids=[d.doc_id for d in all_docs],
                reasoning="No specific matches found, searching broadly.",
                doc_nodes=all_docs,
            )

        # Filter out parent-only grouping nodes (no tree stored)
        parent_ids = {
            d.parent_doc_id for d in candidates.values() if d.parent_doc_id
        }
        searchable = {
            did: d for did, d in candidates.items() if did not in parent_ids
        }
        if not searchable:
            searchable = candidates  # safety fallback

        # 3. Rank candidates with LLM
        ranked = self._rank_candidates(question, list(searchable.values()))

        return ranked

    def _extract_terms(
        self,
        question: str,
        history: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        history_block = ""
        if history:
            recent = history[-4:]
            history_block = "Recent conversation:\n" + "\n".join(
                f"{m['role'].capitalize()}: {m['content']}" for m in recent
            )

        prompt = EXTRACT_SEARCH_TERMS.format(
            question=question,
            history_block=history_block,
        )
        try:
            return chat_json(
                prompt,
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=0.0,
                max_tokens=1024,
                quiet=self.quiet,
                enable_thinking=self.enable_thinking,
                num_ctx=self.num_ctx,
            )
        except Exception as e:
            log.warning(f"Term extraction failed: {e}")
            # Fallback: split the question into words
            words = [w for w in question.split() if len(w) > 3]
            return {"search_terms": words[:5], "topics": words[:3]}

    def _rank_candidates(
        self,
        question: str,
        candidates: List[DocNode],
    ) -> PlanResult:
        # Format document summaries for the LLM
        summaries = []
        for doc in candidates:
            topics_str = ", ".join(doc.topics[:5]) if doc.topics else "N/A"
            summaries.append(
                f"- doc_id: \"{doc.doc_id}\"\n"
                f"  title: \"{doc.title}\"\n"
                f"  summary: \"{doc.summary}\"\n"
                f"  topics: [{topics_str}]"
            )

        prompt = RANK_DOCUMENTS.format(
            question=question,
            doc_summaries="\n".join(summaries),
            max_docs=self.max_docs,
            example_id=candidates[0].doc_id if candidates else "doc_id",
        )
        try:
            result = chat_json(
                prompt,
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=0.0,
                max_tokens=1024,
                quiet=self.quiet,
                enable_thinking=self.enable_thinking,
                num_ctx=self.num_ctx,
            )
            selected_ids = result.get("doc_ids", [])
            reasoning = result.get("reasoning", "")

            # Filter to valid doc_ids
            valid_ids = {d.doc_id for d in candidates}
            final_ids = [did for did in selected_ids if did in valid_ids]

            # If LLM returned nothing useful, take top candidates
            if not final_ids:
                final_ids = [d.doc_id for d in candidates[: self.max_docs]]

            selected_nodes = [
                d for d in candidates if d.doc_id in final_ids
            ]

            return PlanResult(
                doc_ids=final_ids[:self.max_docs],
                reasoning=reasoning,
                doc_nodes=selected_nodes,
            )

        except Exception as e:
            log.warning(f"Document ranking failed: {e}")
            top = candidates[: self.max_docs]
            return PlanResult(
                doc_ids=[d.doc_id for d in top],
                reasoning="Ranking failed, using top candidates.",
                doc_nodes=top,
            )
