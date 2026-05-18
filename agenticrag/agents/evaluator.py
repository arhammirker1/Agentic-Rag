"""
evaluator.py — Evaluator Agent (Iterative Retrieval Gatekeeper).

The Evaluator sits between the Synthesizer and Critic.  It inspects
the draft answer + evidence and decides whether more context is needed.

If the evidence is insufficient, it returns:
  - What specific gaps exist
  - A refined search query for the next Hunter round
  - Whether to expand to additional documents

This enables an iterative retrieval loop:
  Hunt → Synthesize → Evaluate → (loop or proceed to Critic)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..groq_client import chat_json

log = logging.getLogger(__name__)


EVALUATE_PROMPT = """\
You are a retrieval quality evaluator.  You must decide whether the
retrieved evidence is sufficient to fully and accurately answer the
user's question.

Question: "{question}"

Draft answer:
{draft}

Evidence chunks used ({n_chunks} chunks from {n_docs} document(s)):
{evidence_summary}

Documents available but NOT yet searched: {unsearched_docs}

---
Evaluate the draft answer:

1. Does the draft FULLY answer the question with specific details?
2. Are there claims in the draft that lack supporting evidence?
3. Are there obvious aspects of the question left unanswered?
4. Could searching additional documents or nodes yield better information?

Return JSON:
{{
  "sufficient": true or false,
  "confidence": <float 0.0 to 1.0>,
  "gaps": ["<specific gap 1>", "<specific gap 2>"],
  "refined_query": "<a more specific search query to fill the gaps>",
  "expand_docs": true or false
}}

Rules:
- Set "sufficient" to true if the answer covers the question well enough,
  even if not perfectly exhaustive.
- Set "sufficient" to false if the answer says "insufficient evidence",
  is largely empty, or misses key aspects the user asked about.
- "gaps" should list SPECIFIC missing information, not vague complaints.
- "refined_query" should be a concrete, focused question to find the missing info.
- "expand_docs" should be true only if you believe unsearched documents
  might contain the missing information.
- If most nodes have already been visited (high coverage), prefer "sufficient": true
  unless the answer is clearly incomplete — the corpus may simply not contain
  the missing information.
"""


@dataclass
class EvalResult:
    """
    Output of the Evaluator Agent.

    Attributes
    ----------
    sufficient     : Whether the current evidence is enough.
    confidence     : How confident the evaluator is (0-1).
    gaps           : List of specific information gaps.
    refined_query  : A refined search query for the next round.
    expand_docs    : Whether to search additional documents.
    """
    sufficient:    bool       = True
    confidence:    float      = 1.0
    gaps:          List[str]  = field(default_factory=list)
    refined_query: str        = ""
    expand_docs:   bool       = False


class EvaluatorAgent:
    """
    Decides whether a draft answer has sufficient evidence or needs
    another retrieval round.

    Parameters
    ----------
    model   : Groq model ID.
    api_key : Groq API key.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        quiet: bool = False,
        enable_thinking: bool = False,
        num_ctx: int = 32768,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.quiet = quiet
        self.enable_thinking = enable_thinking
        self.num_ctx = num_ctx

    def evaluate(
        self,
        question: str,
        draft: str,
        chunks: List[Dict[str, Any]],
        searched_doc_ids: List[str],
        all_doc_ids: Optional[List[str]] = None,
        total_nodes: int = 0,
        visited_nodes: int = 0,
    ) -> EvalResult:
        """
        Evaluate whether the draft answer has sufficient evidence.

        Parameters
        ----------
        question        : The user's original question.
        draft           : The synthesized draft answer.
        chunks          : The evidence chunks used.
        searched_doc_ids: Doc IDs that have been searched so far.
        all_doc_ids     : All available doc IDs in the forest.
        total_nodes     : Total nodes across all searched trees.
        visited_nodes   : Number of nodes already visited.

        Returns
        -------
        EvalResult with sufficiency decision and gap analysis.
        """
        if not chunks:
            return EvalResult(
                sufficient=False,
                confidence=0.0,
                gaps=["No evidence chunks were retrieved"],
                refined_query=question,
                expand_docs=True,
            )

        # ── Corpus exhaustion check ──────────────────────────────────
        # If we've visited ≥80% of all nodes, the corpus is exhausted.
        # Further rounds will almost certainly find nothing new.
        if total_nodes > 0 and visited_nodes > 0:
            coverage = visited_nodes / total_nodes
            if coverage >= 0.80:
                log.info(
                    f"Corpus exhausted: {visited_nodes}/{total_nodes} "
                    f"nodes visited ({coverage:.0%}) — marking sufficient."
                )
                return EvalResult(
                    sufficient=True,
                    confidence=0.70,
                    gaps=[],
                    refined_query="",
                    expand_docs=False,
                )

        # Build evidence summary (compact — don't send full text)
        evidence_summary = self._summarize_evidence(chunks)

        # Identify unsearched documents
        all_ids = set(all_doc_ids or [])
        searched = set(searched_doc_ids)
        unsearched = list(all_ids - searched)
        unsearched_str = ", ".join(unsearched[:10]) if unsearched else "None"

        # Count unique docs in chunks
        doc_ids_in_chunks = set(c.get("doc_id", "") for c in chunks)

        prompt = EVALUATE_PROMPT.format(
            question=question,
            draft=draft[:3000],
            n_chunks=len(chunks),
            n_docs=len(doc_ids_in_chunks),
            evidence_summary=evidence_summary,
            unsearched_docs=unsearched_str,
        )

        try:
            result = chat_json(
                prompt,
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=0.0,
                max_tokens=512,
                quiet=self.quiet,
                enable_thinking=self.enable_thinking,
                num_ctx=self.num_ctx,
            )
            return EvalResult(
                sufficient=bool(result.get("sufficient", True)),
                confidence=float(result.get("confidence", 0.5)),
                gaps=result.get("gaps", []),
                refined_query=result.get("refined_query", question),
                expand_docs=bool(result.get("expand_docs", False)),
            )
        except Exception as e:
            log.warning(f"Evaluation failed: {e}")
            # On error, assume sufficient to avoid infinite loops
            return EvalResult(sufficient=True, confidence=0.5)

    @staticmethod
    def _summarize_evidence(chunks: List[Dict[str, Any]]) -> str:
        """Create a compact summary of evidence for the evaluator prompt."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            doc_title = chunk.get("doc_title", chunk.get("doc_id", "?"))[:60]
            node_title = chunk.get("node_title", "")[:40]
            text = chunk.get("text", "")
            text_preview = text[:2000] + "..." if len(text) > 2000 else text
            parts.append(
                f"[{i}] Doc: \"{doc_title}\" | Section: \"{node_title}\"\n"
                f"    Preview: {text_preview}"
            )
        return "\n".join(parts)
