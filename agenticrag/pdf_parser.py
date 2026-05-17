"""
pdf_parser.py — Extract per-page text from a PDF or Markdown file.

Tries pdfplumber first (pure-Python), falls back to PyMuPDF if available.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List


def extract_pages(path: str | Path) -> List[str]:
    """
    Extract text from a PDF or Markdown file, one string per page/section.

    For PDFs  → returns one string per physical page.
    For .md   → returns the full text as a single-element list
                (tree building is done by heading structure instead).

    Raises FileNotFoundError if the path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix in (".md", ".markdown", ".txt"):
        return [path.read_text(encoding="utf-8")]

    if suffix == ".pdf":
        return _extract_pdf(path)

    raise ValueError(f"Unsupported file type: {suffix}. Supported: .pdf, .md, .txt")


def convert_pdf_to_markdown(path: str | Path) -> str:
    """
    Convert a PDF to well-structured Markdown using local heuristics.

    Uses pdfplumber's character-level font metadata (size, bold, position)
    to detect headings, tables, bullet lists, and figure captions.
    Produces clean, hierarchical Markdown — no LLM calls required.

    Returns the full document as a single Markdown string.
    """
    from .pdf_to_markdown import convert_pdf_to_markdown as _convert
    return _convert(str(path))


def convert_to_markdown(path: str | Path) -> str:
    """
    DEPRECATED: Use convert_pdf_to_markdown() instead.

    Legacy wrapper around Microsoft MarkItDown.
    """
    try:
        from markitdown import MarkItDown
    except ImportError:
        raise ImportError(
            "The `markitdown` package is required for PDF->Markdown conversion.\n"
            "Install it with:  pip install markitdown[pdf]"
        )

    md = MarkItDown(enable_plugins=False)
    result = md.convert(str(path))
    return result.text_content


def count_tokens(text: str) -> int:
    """
    Approximate token count.
    Uses tiktoken if available, otherwise falls back to char/4.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return max(1, len(text) // 4)


# ── PDF backends ──────────────────────────────────────────────────────────

def _extract_pdf(path: Path) -> List[str]:
    try:
        return _pdfplumber(path)
    except ImportError:
        pass
    try:
        return _pymupdf(path)
    except ImportError:
        pass
    raise ImportError(
        "No PDF library found.  Install one:\n"
        "  pip install pdfplumber      ← recommended\n"
        "  pip install PyMuPDF         ← alternative"
    )


def _pdfplumber(path: Path) -> List[str]:
    import pdfplumber  # type: ignore
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            pages.append(_clean(page.extract_text() or ""))
    return pages


def _pymupdf(path: Path) -> List[str]:
    import fitz  # type: ignore
    pages = []
    doc = fitz.open(str(path))
    for page in doc:
        pages.append(_clean(page.get_text("text") or ""))
    doc.close()
    return pages


def _clean(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()