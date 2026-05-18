"""
synthesizer.py — Synthesizer Agent.

Takes the raw text chunks retrieved by the Hunter Agents from multiple
documents and produces a single, cohesive answer with proper citations.

The Synthesizer enforces strict citation rules:
  - Every claim must cite [DocTitle, Pages X-Y].
  - If no evidence supports a claim, it must be omitted.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from ..groq_client import chat

log = logging.getLogger(__name__)


def _deduplicate_chunk_text(text: str) -> str:
    """Remove repeated sentences from chunk text before synthesis."""
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


SYNTHESIZE_PROMPT = """\
You are a research synthesis expert.  You have been given text chunks
retrieved from multiple documents to answer a user's question.

Question: "{question}"

{history_block}

Retrieved evidence:
{evidence}

---
Write a DETAILED and COMPREHENSIVE answer based ONLY on the evidence above.

STRICT RULES:
1. Every factual claim MUST cite its source using [Source: <doc_title>, Pages <start>-<end>].
2. If a claim cannot be backed by any retrieved chunk, DO NOT include it.
3. If the evidence is insufficient to answer, say so clearly.
4. Synthesise information across documents — don't just list per-document summaries.
5. Use clear, professional language.
6. If documents disagree, present both perspectives with citations.
7. DO NOT merely list section names or topic labels — EXPLAIN what each point actually says.
   For example, instead of "the document mentions Security & Authentication",
   write "The document identifies Security & Authentication as a key risk,
   noting that storing API keys requires encrypted vaults and restricted scopes [Source: ...]".
8. Include specific details, numbers, examples, and explanations from the evidence.
9. Structure your answer with clear paragraphs or bullet points for readability.
10. Aim for a thorough answer that fully addresses the question using ALL relevant evidence.
"""


class SynthesizerAgent:
    """
    Combines findings from multiple Hunter Agents into a cohesive answer.

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
        max_output_tokens: int = 4096,
        max_chunk_size: int = 2000,
        max_evidence_size: int = 24000,
        table_parsing_mode: bool = True,
        quiet: bool = False,
        enable_thinking: bool = False,
        num_ctx: int = 32768,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.max_output_tokens = max_output_tokens
        self.max_chunk_size = max_chunk_size
        self.max_evidence_size = max_evidence_size
        self.table_parsing_mode = table_parsing_mode
        self.quiet = quiet
        self.enable_thinking = enable_thinking
        self.num_ctx = num_ctx

    @staticmethod
    def _contains_table(text: str) -> bool:
        """
        Return True when *text* contains at least one Markdown table.

        Detection heuristic:
          1. Count lines that contain the pipe character (|) — need ≥ 2.
          2. At least one of those lines must match a separator row pattern
             such as ``| --- | :--- | ---: |``.

        This reliably identifies pdfplumber-extracted tables without
        false-positives on prose that happens to contain a single pipe.
        """
        lines = text.splitlines()
        pipe_lines = [ln for ln in lines if "|" in ln]
        if len(pipe_lines) < 2:
            return False
        separator_re = re.compile(r"^\|?[\s\-:]+(\|[\s\-:]+)+\|?\s*$")
        return any(separator_re.match(ln) for ln in pipe_lines)

    def synthesize(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Produce a synthesized answer from multiple document chunks.

        Parameters
        ----------
        question : The user's question.
        chunks   : List of chunk dicts from HunterAgent, each with:
                   doc_id, doc_title, node_title, start_page, end_page, text.
        history  : Optional conversation history.

        Returns
        -------
        The synthesized answer string with citations.
        """
        if not chunks:
            return (
                "I could not find any relevant information in the indexed documents "
                "to answer this question."
            )

        # Format evidence
        evidence_parts = []
        for i, chunk in enumerate(chunks, 1):
            doc_title = chunk.get("doc_title", chunk.get("doc_id", "Unknown"))
            node_title = chunk.get("node_title", "")
            start = chunk.get("start_page", "?")
            end = chunk.get("end_page", "?")
            text = chunk.get("text", "")

            # Deduplicate repeated sentences within each chunk
            text = _deduplicate_chunk_text(text)

            # Cap individual chunk length to prevent token overflow.
            # Exception: when table_parsing_mode is active and the chunk
            # contains a Markdown table we send the full text so that no
            # rows are silently dropped (e.g. executive officer lists,
            # financial schedules).  The keyword pre-filter already
            # narrows the evidence set to a handful of nodes, so the
            # token budget impact is acceptable.
            if self.table_parsing_mode and self._contains_table(text):
                pass  # preserve the complete table
            elif self.max_chunk_size is not None and len(text) > self.max_chunk_size:
                text = text[:self.max_chunk_size] + " …[truncated]"

            header = f"[Chunk {i}] Document: \"{doc_title}\""
            if node_title:
                header += f" | Section: \"{node_title}\""
            header += f" | Pages: {start}-{end}"

            evidence_parts.append(f"{header}\n{text}")

        evidence = "\n\n---\n\n".join(evidence_parts)

        # Cap total evidence length
        if self.max_evidence_size is not None and len(evidence) > self.max_evidence_size:
            evidence = evidence[:self.max_evidence_size] + "\n\n…[evidence truncated]"

        # Format history
        history_block = ""
        if history:
            recent = history[-4:]
            history_block = "Conversation history:\n" + "\n".join(
                f"{m['role'].capitalize()}: {m['content']}" for m in recent
            )

        prompt = SYNTHESIZE_PROMPT.format(
            question=question,
            evidence=evidence,
            history_block=history_block,
        )

        try:
            return chat(
                prompt,
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=0.1,
                max_tokens=self.max_output_tokens,
                quiet=self.quiet,
                enable_thinking=self.enable_thinking,
                num_ctx=self.num_ctx,
            ).strip()
        except Exception as e:
            log.error(f"Synthesis failed: {e}")
            return f"[Error generating synthesized answer: {e}]"
