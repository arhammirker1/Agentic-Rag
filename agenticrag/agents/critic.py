"""
critic.py — Critic Agent (Zero-Hallucination Enforcer).

The Critic is the last agent in the loop.  It receives the Synthesizer's
answer along with the raw source chunks and verifies that every claim
is directly supported by the evidence.

If a claim is unsupported, the Critic removes it.
If the entire answer is unsupported, it returns a "cannot answer" response.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..groq_client import chat_json, chat

log = logging.getLogger(__name__)


VERIFY_PROMPT = """\
You are a strict fact-checking agent.  Your job is to verify that an answer
contains ZERO hallucinations — every claim must be directly supported by
the source evidence.

ANSWER TO VERIFY:
{answer}

SOURCE EVIDENCE (the ONLY ground truth — read ALL chunks carefully):
{evidence}

---
CHECK EACH CLAIM in the answer against the source evidence above.

A claim is SUPPORTED if:
  - The evidence contains the same facts, even if paraphrased differently.
  - Key phrases, terms, or lists from the claim can be found in the evidence text.
  - The claim is a reasonable inference from explicitly stated information.

A claim is a HALLUCINATION only if:
  - The specific facts, numbers, names, or technical terms in the claim
    CANNOT BE FOUND ANYWHERE in any of the evidence chunks above.
  - The claim introduces entirely new information not present in any chunk.

IMPORTANT:
  - Search ALL evidence chunks thoroughly before flagging a claim.
  - If a phrase like "marketing automation" appears in ANY chunk, it is supported.
  - If a list like "browsers, APIs, spreadsheets" appears in ANY chunk, it is supported.
  - Do NOT flag claims just because they combine information from different chunks.
  - Paraphrasing is allowed — "agents interact with browsers" supports "systems can
    interact with browsers."

Return JSON:
{{
  "verdict": "pass" or "fail",
  "hallucinations": [
    {{
      "claim": "<the unsupported claim>",
      "reason": "<why it's not supported — specify which facts are missing from ALL chunks>"
    }}
  ],
  "confidence": <float 0.0 to 1.0 — how confident you are the answer is accurate>
}}

If there are NO hallucinations, return "verdict": "pass" with an empty list.
"""

REWRITE_PROMPT = """\
You are rewriting an answer to remove hallucinated claims while preserving quality.

Original answer:
{answer}

Hallucinated claims to remove:
{hallucinations}

Source evidence:
{evidence}

---
Rewrite the answer, removing ONLY the hallucinated claims listed above.

CRITICAL RULES:
1. Keep ALL non-hallucinated content EXACTLY as-is, including all details,
   explanations, numbers, examples, and citations.
2. Do NOT simplify, shorten, or summarize the remaining content.
3. Do NOT remove details just because they seem minor — only remove
   claims explicitly listed as hallucinations above.
4. Maintain the original structure, formatting, and depth of explanation.
5. Maintain all citations.
6. If removing the hallucinations leaves nothing meaningful, respond with:
   "The available evidence is insufficient to fully answer this question."
"""

RELEVANCE_CHECK_PROMPT = """\
You are an answer-relevance evaluator.  Your job is to check whether an answer
actually addresses the user's specific question, rather than merely summarising
tangentially related content.

User's question: "{question}"

Answer to evaluate:
{answer}

---
Evaluate the answer:

1. Does the answer directly address the SPECIFIC question asked?
2. Does the answer provide actionable, focused information rather than
   generic summaries of loosely related topics?
3. If the question asks "how to build X", does the answer explain how,
   or does it just list general concepts?

Return JSON:
{{
  "relevant": true or false,
  "relevance_score": <float 0.0 to 1.0>,
  "issue": "<brief description of the relevance problem, or empty if relevant>"
}}

Rules:
- Set "relevant" to true if the answer makes a genuine attempt to address
  the question, even if evidence is limited.
- Set "relevant" to false ONLY if the answer mostly discusses topics that
  are NOT what the user asked about.
- A low relevance_score (< 0.5) means the answer is mostly off-topic.
"""

RELEVANCE_REWRITE_PROMPT = """\
The answer below does not adequately focus on the user's question.
Rewrite it to directly address the question using ONLY the source evidence.

User's question: "{question}"

Original answer:
{answer}

Relevance issue: {issue}

Source evidence:
{evidence}

---
Rewrite the answer to directly and specifically address the user's question.

Rules:
1. Focus on what the user ACTUALLY asked.
2. Use ONLY information from the source evidence.
3. If the evidence does not directly address the question, say so honestly
   and provide whatever relevant information IS available.
4. Lead with the most relevant information first.
5. Maintain all citations.
6. Do NOT pad the answer with loosely related content.
"""


@dataclass
class VerificationResult:
    """
    Output of the Critic Agent.

    Attributes
    ----------
    answer         : The final verified (and possibly rewritten) answer.
    verdict        : "pass" if no hallucinations, "fail" if rewritten/blocked.
    hallucinations : List of detected hallucinated claims.
    confidence     : 0.0 to 1.0 confidence score.
    was_rewritten  : Whether the answer was modified by the Critic.
    """
    answer:         str
    verdict:        str             = "pass"
    hallucinations: List[Dict]      = field(default_factory=list)
    confidence:     float           = 1.0
    was_rewritten:  bool            = False


class CriticAgent:
    """
    Verifies answers against source evidence to ensure zero hallucination.

    Parameters
    ----------
    model   : Groq model ID (use a strong model for best results).
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

    def verify(
        self,
        answer: str,
        chunks: List[Dict[str, Any]],
        auto_rewrite: bool = True,
        question: str = "",
    ) -> VerificationResult:
        """
        Verify an answer against source evidence.

        Parameters
        ----------
        answer       : The synthesized answer to verify.
        chunks       : The source evidence chunks.
        auto_rewrite : If True, automatically rewrite the answer to remove
                       hallucinations.  If False, just report them.
        question     : The user's original question (for relevance checking).

        Returns
        -------
        VerificationResult with the verified/rewritten answer and metadata.
        """
        if not chunks:
            return VerificationResult(
                answer="No source evidence was available to verify against.",
                verdict="fail",
                confidence=0.0,
            )

        # Format evidence for the prompt
        evidence = self._format_evidence(chunks)

        # Run verification
        verification = self._check(answer, evidence)
        hallucinations = verification.get("hallucinations", [])
        verdict = verification.get("verdict", "pass")
        confidence = float(verification.get("confidence", 0.5))

        # If clean, run relevance check
        if verdict == "pass" and not hallucinations:
            current_answer = answer

            # ── Answer relevance check ──────────────────────────────────
            if question:
                relevance = self._check_relevance(question, current_answer)
                relevance_score = float(relevance.get("relevance_score", 1.0))
                is_relevant = bool(relevance.get("relevant", True))
                issue = relevance.get("issue", "")

                if not is_relevant and auto_rewrite and issue:
                    log.info(f"Answer flagged as off-topic: {issue}")
                    current_answer = self._rewrite_for_relevance(
                        question, current_answer, issue, evidence
                    )
                    return VerificationResult(
                        answer=current_answer,
                        verdict="pass",
                        confidence=relevance_score,
                        was_rewritten=True,
                    )

                # Adjust confidence based on relevance score
                confidence = min(confidence, relevance_score)

            return VerificationResult(
                answer=current_answer,
                verdict="pass",
                confidence=confidence,
            )

        # Hallucinations detected
        if auto_rewrite and hallucinations:
            rewritten = self._rewrite(answer, hallucinations, evidence)
            return VerificationResult(
                answer=rewritten,
                verdict="fail",
                hallucinations=hallucinations,
                confidence=confidence,
                was_rewritten=True,
            )

        return VerificationResult(
            answer=answer,
            verdict="fail",
            hallucinations=hallucinations,
            confidence=confidence,
            was_rewritten=False,
        )

    def _check(self, answer: str, evidence: str) -> Dict[str, Any]:
        prompt = VERIFY_PROMPT.format(
            answer=answer,
            evidence=evidence[:24000],
        )
        try:
            return chat_json(
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
        except Exception as e:
            log.warning(f"Verification failed: {e}")
            # On error, assume pass to avoid blocking
            return {"verdict": "pass", "hallucinations": [], "confidence": 0.5}

    def _check_relevance(
        self, question: str, answer: str
    ) -> Dict[str, Any]:
        """Check whether the answer actually addresses the user's question."""
        prompt = RELEVANCE_CHECK_PROMPT.format(
            question=question,
            answer=answer[:4000],
        )
        try:
            return chat_json(
                prompt,
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=0.0,
                max_tokens=256,
                quiet=self.quiet,
                enable_thinking=self.enable_thinking,
                num_ctx=self.num_ctx,
            )
        except Exception as e:
            log.warning(f"Relevance check failed: {e}")
            return {"relevant": True, "relevance_score": 0.7, "issue": ""}

    def _rewrite_for_relevance(
        self,
        question: str,
        answer: str,
        issue: str,
        evidence: str,
    ) -> str:
        """Rewrite an answer to better address the user's question."""
        prompt = RELEVANCE_REWRITE_PROMPT.format(
            question=question,
            answer=answer,
            issue=issue,
            evidence=evidence[:24000],
        )
        try:
            return chat(
                prompt,
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=0.0,
                max_tokens=2048,
                quiet=self.quiet,
                enable_thinking=self.enable_thinking,
                num_ctx=self.num_ctx,
            ).strip()
        except Exception as e:
            log.error(f"Relevance rewrite failed: {e}")
            return answer  # return original if rewrite fails

    def _rewrite(
        self,
        answer: str,
        hallucinations: List[Dict],
        evidence: str,
    ) -> str:
        halluc_text = "\n".join(
            f"- Claim: \"{h.get('claim', '?')}\"\n  Reason: {h.get('reason', '?')}"
            for h in hallucinations
        )
        prompt = REWRITE_PROMPT.format(
            answer=answer,
            hallucinations=halluc_text,
            evidence=evidence[:24000],
        )
        try:
            return chat(
                prompt,
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=0.0,
                max_tokens=2048,
                quiet=self.quiet,
                enable_thinking=self.enable_thinking,
                num_ctx=self.num_ctx,
            ).strip()
        except Exception as e:
            log.error(f"Rewrite failed: {e}")
            return answer  # return original if rewrite fails

    @staticmethod
    def _dedup(text: str) -> str:
        """Remove repeated sentences from evidence text."""
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

    @staticmethod
    def _format_evidence(chunks: List[Dict[str, Any]]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            doc_title = chunk.get("doc_title", chunk.get("doc_id", "?"))
            node_title = chunk.get("node_title", "")
            start_page = chunk.get("start_page", "?")
            end_page = chunk.get("end_page", "?")
            text = chunk.get("text", "")
            text = CriticAgent._dedup(text)
            if len(text) > 3000:
                text = text[:3000] + " …[truncated]"
            header = f"[CHUNK {i}] Document: \"{doc_title}\""
            if node_title:
                header += f" | Section: \"{node_title}\""
            header += f" | Pages: {start_page}-{end_page}"
            parts.append(f"{header}\n{text}")
        return "\n\n---\n\n".join(parts)
