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


# ── Sub-tree metadata (local, no LLM) ────────────────────────────────────

def extract_subtree_metadata(
    nodes: list,
    parent_title: str = "",
    source_file: str = "",
) -> Dict[str, Any]:
    """
    Extract metadata for a sub-tree locally (no LLM call).

    Uses the node titles and text content to build a rich summary
    and topic list.  This is fast and doesn't hit rate limits.

    Parameters
    ----------
    nodes       : The sub-tree's node list.
    parent_title: Title of the parent document.
    source_file : Original filename.

    Returns
    -------
    Dict with title, summary, topics, entities.
    """
    # Collect all titles and text from the sub-tree
    titles: List[str] = []
    all_text: List[str] = []
    _collect_content(nodes, titles, all_text)

    # Build title from section headings
    if len(titles) == 1:
        title = titles[0]
    elif len(titles) <= 3:
        title = " | ".join(titles[:3])
    else:
        title = f"{titles[0]} ... {titles[-1]} ({len(titles)} sections)"

    if parent_title:
        title = f"{parent_title} — {title}"

    # Build summary from first ~500 chars of combined text
    combined_text = " ".join(all_text)
    summary_text = combined_text[:500].strip()
    if len(combined_text) > 500:
        dot = summary_text.rfind(".")
        if dot > 200:
            summary_text = summary_text[:dot + 1]
        else:
            summary_text += "..."

    summary = f"Sections: {', '.join(titles[:8])}. {summary_text}"

    # Extract topics from titles
    topics = _extract_topics_from_titles(titles)

    return {
        "title": title,
        "summary": summary,
        "topics": topics,
        "entities": [],
        "doc_type_hint": "section",
    }


def _collect_content(
    nodes: list,
    titles: List[str],
    texts: List[str],
) -> None:
    """Recursively collect titles and text snippets from nodes."""
    for n in nodes:
        t = n.get("title", "").strip()
        if t:
            titles.append(t)
        text = n.get("text", "").strip()
        if text:
            texts.append(text[:200])
        if n.get("nodes"):
            _collect_content(n["nodes"], titles, texts)


def _extract_topics_from_titles(titles: list) -> List[str]:
    """Extract topic keywords from section titles."""
    stop = {
        "the", "a", "an", "of", "in", "to", "for", "and", "or", "is",
        "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should", "may",
        "might", "shall", "can", "with", "at", "by", "from", "on", "not",
        "no", "but", "if", "then", "than", "too", "very", "just", "about",
        "this", "that", "these", "those", "it", "its", "our", "we", "us",
        "page", "part", "item", "section", "general", "content",
    }
    words: Dict[str, int] = {}
    for title in titles:
        for word in title.split():
            clean = word.strip(".,;:()[]{}\"'").lower()
            if len(clean) > 2 and clean not in stop:
                words[clean] = words.get(clean, 0) + 1

    sorted_words = sorted(words.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:10]]
