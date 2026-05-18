"""
keyword_agent.py — Keyword Expansion Agent.

Translates a user's natural-language question into a rich set of keyphrases,
keywords, and synonyms for use in the local node pre-filtering step inside
large document trees (the Hybrid Sub-Tree Filtering pipeline).

Responsibilities:
  - Call the LLM once with EXPAND_KEYWORDS to produce structured search terms.
  - Validate and sanitise the JSON response.
  - Fall back to a local regex-based extractor if the LLM call fails (rate-limit,
    timeout, bad JSON) so the RAG pipeline never crashes.

Why a dedicated agent instead of an inline call:
  - Separation of concerns: TreeSearcher does tree navigation; this agent
    handles API orchestration, JSON validation, and failure recovery.
  - Isolated testability: keyword generation can be unit-tested independently.
  - Swap-ability: replace the LLM expander with KeyBERT or spaCy by editing
    only this one file.
  - Consistency: matches PlannerAgent / HunterAgent / SynthesizerAgent pattern.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from ..groq_client import chat_json
from ..prompts import EXPAND_KEYWORDS

log = logging.getLogger(__name__)


class KeywordAgent:
    """
    Generates expanded search keywords from a user question.

    Used by TreeSearcher to pre-filter large document trees before the
    agentic SELECT_NODES loop, reducing prompt token usage by ~95%.

    Parameters
    ----------
    model            : LLM model ID.
    api_key          : API key (falls back to GROQ_API_KEY env var).
    base_url         : Custom endpoint for local LLMs (Ollama, vLLM, etc.).
    quiet            : Suppress console output.
    enable_thinking  : Enable deep thinking mode (Qwen3 / DeepSeek R1).
    num_ctx          : Context window size for local LLMs.
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
        self.model           = model
        self.api_key         = api_key
        self.base_url        = base_url
        self.quiet           = quiet
        self.enable_thinking = enable_thinking
        self.num_ctx         = num_ctx

    def expand(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> List[str]:
        """
        Return a flat, deduplicated list of lowercase search terms for
        ``question``, combining LLM-generated expansions with raw question
        words as a safety-net.

        Falls back gracefully to local extraction if the LLM call fails.

        Parameters
        ----------
        question : The user's question.
        history  : Optional conversation history for additional context.

        Returns
        -------
        List of lowercase keyword strings, best-expanded terms first.
        """
        from ..utils.logging import trail

        trail.step(
            "KEYWORD AGENT (INPUT)",
            f"Expanding keywords for question: '{question}'",
            {"question": question, "has_history": bool(history)},
            quiet=self.quiet
        )

        history = history or []

        history_block = ""
        if history:
            recent = history[-4:]
            history_block = "Conversation history:\n" + "\n".join(
                f"{m['role'].capitalize()}: {m['content']}" for m in recent
            )

        prompt = EXPAND_KEYWORDS.format(
            question=question,
            history_block=history_block,
        )

        try:
            result = chat_json(
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
            keywords = self._parse_result(result)

            # Always append raw question words so critical terms from the
            # original query are never missed even if the LLM omits them.
            keywords.extend(w.lower() for w in question.split() if len(w) > 3)

            final_keywords = self._deduplicate(keywords)
            trail.step(
                "KEYWORD AGENT (SUCCESS)",
                f"Expanded search keywords generated successfully.",
                {"expanded_keywords": final_keywords},
                quiet=self.quiet
            )
            return final_keywords

        except Exception as e:
            log.warning(
                f"KeywordAgent LLM call failed ({e}) — "
                f"falling back to local extraction."
            )
            fallback_kws = self._local_fallback(question)
            trail.step(
                "KEYWORD AGENT (FALLBACK)",
                f"LLM call failed ({e}). Falling back to local heuristic extraction.",
                {"fallback_keywords": fallback_kws},
                quiet=self.quiet
            )
            return fallback_kws

    # ── Private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _parse_result(result: Any) -> List[str]:
        """
        Extract and flatten keywords from the LLM JSON response.
        Handles missing keys and non-string values gracefully.
        """
        if not isinstance(result, dict):
            return []
        keywords: List[str] = []
        for field in ("keyphrases", "keywords", "synonyms"):
            items = result.get(field, [])
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, str) and item.strip():
                        keywords.append(item.lower().strip())
        return keywords

    @staticmethod
    def _deduplicate(keywords: List[str]) -> List[str]:
        """Remove duplicates while preserving insertion order."""
        seen: set = set()
        out: List[str] = []
        for kw in keywords:
            if kw and kw not in seen:
                seen.add(kw)
                out.append(kw)
        return out

    @staticmethod
    def _local_fallback(question: str) -> List[str]:
        """
        Lightweight local keyword extractor used when the LLM call fails.

        Extracts candidate keywords by lowercasing, tokenising on word
        boundaries, removing common English stop-words, and keeping tokens
        of 4+ characters.  Guarantees the pre-filter always has something
        to work with even under complete API outage.
        """
        _STOP_WORDS = {
            "this", "that", "with", "from", "have", "will", "been",
            "they", "their", "there", "what", "which", "when", "where",
            "about", "would", "could", "should", "into", "over", "then",
            "than", "also", "each", "were", "your", "more", "some",
            "does", "just", "like", "very", "only", "such", "both",
            "these", "those", "them", "being",
        }
        tokens = re.findall(r"[a-z]{4,}", question.lower())
        return [t for t in dict.fromkeys(tokens) if t not in _STOP_WORDS]