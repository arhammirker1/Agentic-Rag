"""
metadata.py — LLM-powered metadata extraction for document ingestion.

When a document is added to the Forest, this module extracts structured
metadata (title, topics, entities, summary) from its raw text using the LLM.
This metadata powers the Planner Agent's document selection.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..groq_client import chat_json

log = logging.getLogger(__name__)


# ── Prompt ────────────────────────────────────────────────────────────────

EXTRACT_METADATA_PROMPT = """\
You are a metadata extraction expert.  Given the first few pages of a document,
extract structured metadata.

Document text (first ~3000 chars):
{text}

---
Extract the following and return as JSON:
{{
  "title": "<document title — infer from content if not explicit>",
  "summary": "<2-3 sentence summary of what this document covers>",
  "topics": ["<topic1>", "<topic2>", ...],
  "entities": ["<entity1>", "<entity2>", ...],
  "doc_type_hint": "<report|contract|manual|paper|presentation|other>"
}}

Rules:
- "topics" should be 3-10 broad themes/keywords (e.g. "revenue", "AI strategy", "compliance").
- "entities" should be specific named entities: people, organisations, products, dates, etc.
- Keep the summary factual and concise.
- If the text is too short to extract something, use an empty list or empty string.
"""


def extract_metadata(
    pages: List[str],
    *,
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_chars: int = 3000,
    quiet: bool = False,
    enable_thinking: bool = False,
    num_ctx: int = 32768,
) -> Dict[str, Any]:
    """
    Extract structured metadata from document pages using the LLM.

    Parameters
    ----------
    pages     : List of page strings from the document.
    model     : Groq model ID.
    api_key   : Groq API key.
    max_chars : Max characters to send to the LLM (from the start of the doc).

    Returns
    -------
    Dict with keys: title, summary, topics, entities, doc_type_hint.
    """
    # Combine first pages up to max_chars
    combined = ""
    for page in pages:
        if len(combined) + len(page) > max_chars:
            remaining = max_chars - len(combined)
            if remaining > 100:
                combined += page[:remaining]
            break
        combined += page + "\n\n"

    if not combined.strip():
        return {
            "title": "",
            "summary": "",
            "topics": [],
            "entities": [],
            "doc_type_hint": "other",
        }

    prompt = EXTRACT_METADATA_PROMPT.format(text=combined)

    try:
        result = chat_json(
            prompt,
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0.0,
            max_tokens=512,
            quiet=quiet,
            enable_thinking=enable_thinking,
            num_ctx=num_ctx,
        )
        # Normalise the result
        return {
            "title": result.get("title", ""),
            "summary": result.get("summary", ""),
            "topics": result.get("topics", []),
            "entities": result.get("entities", []),
            "doc_type_hint": result.get("doc_type_hint", "other"),
        }
    except Exception as e:
        log.warning(f"Metadata extraction failed: {e}")
        return {
            "title": "",
            "summary": "",
            "topics": [],
            "entities": [],
            "doc_type_hint": "other",
        }
