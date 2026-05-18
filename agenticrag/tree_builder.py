"""
tree_builder.py — Build the PageIndex tree from a document.

Algorithm (v2 — pdf_to_markdown + Heading Parser)
--------------------------------------------------
1. Convert PDF to structured Markdown using the local pdf_to_markdown
   converter (font-size heading detection, table extraction, bullet
   merging — zero LLM calls).
2. Parse the Markdown into a tree using heading levels.
3. Assign node IDs, generate doc description.
4. Return a JSON-serialisable dict.

Fallback: if the PDF converter fails, falls back to the legacy
page-scanning pipeline (LLM-based TOC detection).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import PageIndexConfig
from .groq_client import chat, chat_json
from .pdf_parser import count_tokens, extract_pages, convert_pdf_to_markdown
from .utils.logging import trail
from .prompts import (
    BUILD_FROM_TOC,
    BUILD_NO_TOC,
    DETECT_TOC,
    DOC_DESCRIPTION,
    MERGE_TREES,
    NODE_SUMMARY,
    SYS_BUILDER,
)

log = logging.getLogger(__name__)


# (No LLM prompts needed — pdf_to_markdown handles structure locally)


# ─── Public API ───────────────────────────────────────────────────────────

def build_tree(
    path: str | Path,
    config: Optional[PageIndexConfig] = None,
) -> Dict[str, Any]:
    """
    Build a PageIndex tree from a PDF, Markdown, or text file.

    Returns
    -------
    dict with keys:
      "document_description" : str   (if config.add_doc_description)
      "source_file"          : str
      "total_pages"          : int
      "nodes"                : list  — the hierarchical tree
    """
    if config is None:
        config = PageIndexConfig()

    path = Path(path)
    _log(config, f"[*] Reading {path.name} ...")

    pages = extract_pages(path)
    total = len(pages)
    _log(config, f"   {total} pages extracted.")

    # ── Choose build strategy ─────────────────────────────────────────
    use_local_md = False

    if path.suffix.lower() in (".md", ".markdown", ".txt"):
        # Markdown files — parse headings directly, zero LLM calls
        nodes = _parse_markdown(pages[0] if pages else "")
        use_local_md = True
    else:
        # PDFs — use the local pdf_to_markdown converter (no LLM)
        try:
            _log(config, "[*] Converting PDF to structured Markdown (local) ...")
            structured_md = convert_pdf_to_markdown(path)
            trail.step("PDF_TO_MARKDOWN", f"Converted {len(structured_md)} chars from {path.name}", structured_md[:1000] + "...")

            nodes = _parse_markdown(structured_md)
            use_local_md = True
            _log(config, f"    Built {_count_nodes(nodes)} nodes from structured Markdown.")
            trail.step("TREE BUILT", f"Created {_count_nodes(nodes)} nodes (zero LLM calls)", nodes)
        except Exception as e:
            log.warning(f"pdf_to_markdown failed ({e}), falling back to legacy pipeline.")
            _log(config, f"    Converter failed: {e}. Using legacy pipeline.")
            nodes = _build_pdf_tree(pages, config)

    # ── Enrich ────────────────────────────────────────────────────────
    if config.add_node_id:
        _number_nodes(nodes)

    if config.add_node_summary and not use_local_md:
        # Local-MD nodes already have full text — summaries are redundant
        _log(config, "[*] Generating node summaries ...")
        _summarise_nodes(nodes, pages, config)

    if config.add_node_text and not use_local_md:
        # Local-MD nodes already have text from heading parsing
        _embed_text(nodes, pages)

    # Assemble result
    result: Dict[str, Any] = {
        "source_file": path.name,
        "total_pages": total,
        "nodes": nodes,
    }

    # Expose raw markdown for the tree splitter (stripped before saving)
    if use_local_md:
        result["_markdown"] = structured_md

    if config.add_doc_description:
        _log(config, "[*] Generating document description ...")
        result["document_description"] = _make_doc_description(nodes, config)

    _log(config, "[OK] Indexing complete.")
    return result


# _refine_markdown_with_llm removed — pdf_to_markdown handles structure locally


# ─── PDF tree building ────────────────────────────────────────────────────

def _build_pdf_tree(
    pages: List[str],
    config: PageIndexConfig,
) -> List[Dict]:
    # 1. Detect TOC
    toc = _detect_toc(pages, config)
    if toc.get("has_toc") and toc.get("toc_page") is not None:
        _log(config, f"   TOC found on page {toc['toc_page']}. Building from TOC …")
        return _from_toc(pages, toc["toc_page"], config)

    _log(config, "   No TOC found. Building by scanning pages …")
    return _by_scanning(pages, config)


def _detect_toc(pages: List[str], config: PageIndexConfig) -> Dict:
    n = min(config.toc_check_pages, len(pages))
    prompt = DETECT_TOC.format(n=n, pages=_fmt_pages(pages[:n]))
    try:
        return chat_json(
            prompt,
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            system=SYS_BUILDER,
            temperature=0.0,
            max_tokens=128,
            quiet=config.quiet,
            enable_thinking=config.enable_thinking,
            num_ctx=config.num_ctx,
        )
    except Exception as e:
        log.debug(f"TOC detection failed: {e}")
        return {"has_toc": False, "toc_page": None}


def _from_toc(
    pages: List[str], toc_page: int, config: PageIndexConfig
) -> List[Dict]:
    prompt = BUILD_FROM_TOC.format(
        toc_page=toc_page,
        toc_text=pages[toc_page],
        total_pages=len(pages),
    )
    try:
        result = chat_json(
            prompt,
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            system=SYS_BUILDER,
            temperature=0.0,
            max_tokens=config.max_output_tokens,
            quiet=config.quiet,
            enable_thinking=config.enable_thinking,
            num_ctx=config.num_ctx,
        )
        if isinstance(result, dict):
            return result.get("nodes", [result])
        return result
    except Exception as e:
        log.warning(f"TOC build failed ({e}), falling back to scan …")
        return _by_scanning(pages, config)


def _by_scanning(pages: List[str], config: PageIndexConfig) -> List[Dict]:
    """Slide a window of pages, build partial subtrees, then merge."""
    partials: List[List[Dict]] = []
    total = len(pages)
    step = config.max_pages_per_node
    next_id = 1

    for start in range(0, total, step):
        end = min(start + step, total)
        window = _fmt_pages(pages[start:end], offset=start)

        # Trim if over token budget
        if count_tokens(window) > config.max_tokens_per_node:
            window = window[: config.max_tokens_per_node * 4]

        prompt = BUILD_NO_TOC.format(
            start=start,
            end=end - 1,
            pages=window,
            next_id=f"{next_id:04d}",
            max_pages=config.max_pages_per_node,
        )
        try:
            partial = chat_json(
                prompt,
                model=config.model,
                api_key=config.api_key,
                base_url=config.base_url,
                system=SYS_BUILDER,
                temperature=0.0,
                max_tokens=config.max_output_tokens,
                quiet=config.quiet,
                enable_thinking=config.enable_thinking,
            num_ctx=config.num_ctx,
            )
            if isinstance(partial, dict):
                partial = [partial]
            partials.append(partial)
            next_id += _count_nodes(partial)
        except Exception as e:
            log.warning(f"Subtree build failed pages {start}-{end}: {e}")
            partials.append([_fallback_node(start, end, next_id)])
            next_id += 1

    if not partials:
        return []
    if len(partials) == 1:
        return partials[0]
    return _merge(partials, config)


def _merge(partials: List[List[Dict]], config: PageIndexConfig) -> List[Dict]:
    flat = [n for sub in partials for n in sub]
    prompt = MERGE_TREES.format(partial=json.dumps(flat, indent=2))
    try:
        merged = chat_json(
            prompt,
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            system=SYS_BUILDER,
            temperature=0.0,
            max_tokens=config.max_output_tokens,
            quiet=config.quiet,
            enable_thinking=config.enable_thinking,
            num_ctx=config.num_ctx,
        )
        return merged if isinstance(merged, list) else flat
    except Exception:
        return flat


# ─── Markdown tree (heading-based, no LLM for structure) ──────────────────

def _deduplicate_text(text: str) -> str:
    """
    Remove repeated sentences/paragraphs from node text.

    Many PDFs (especially from copy-paste or OCR) produce the same
    paragraph repeated 5-10x.  This inflates tokens and causes the
    LLM to produce repetitive filler answers.

    Strategy:
      1. First normalise: insert a space after sentence-ending punctuation
         when it's immediately followed by an uppercase letter (handles
         the common "documents.Memory" concatenation pattern).
      2. Split on sentence boundaries.
      3. Keep only the first occurrence of each sentence.
    """
    if not text or len(text) < 100:
        return text

    # Fix missing whitespace after sentence-ending punctuation
    # e.g. "...documents.Memory" -> "...documents. Memory"
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)

    # Split on sentence-ending punctuation followed by whitespace
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


def _deduplicate_nodes(nodes: List[Dict]) -> None:
    """Apply paragraph deduplication to all nodes in the tree."""
    for n in nodes:
        if n.get("text"):
            n["text"] = _deduplicate_text(n["text"])
        _deduplicate_nodes(n.get("nodes", []))


def _parse_markdown(text: str) -> List[Dict]:
    """
    Parse Markdown text into a tree based on heading levels.
    Each heading (# to ######) becomes a node, with body text
    stored in the node's "text" field.

    Also parses <!-- page:N --> markers emitted by pdf_to_markdown
    to set start_index and end_index on each node for proper citations.
    """
    # Strip LLM code fences if present
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:markdown)?\s*\n?", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\n?```\s*$", "", stripped)

    lines = stripped.split("\n")
    roots: List[Dict] = []
    stack: List[Dict] = []        # active ancestors
    body_lines: List[str] = []
    current: Optional[Dict] = None
    current_page: int = 0         # track current page from markers

    # Regex for page markers: <!-- page:N -->
    page_marker_re = re.compile(r'^\s*<!--\s*page:(\d+)\s*-->\s*$')

    def _flush():
        if current is not None:
            # Filter out page markers from the body text
            clean = [l for l in body_lines if not page_marker_re.match(l)]
            current["text"] = "\n".join(clean).strip()
            # Set end_index to the last page seen in the body
            current["end_index"] = current_page
        body_lines.clear()

    for line in lines:
        # Check for page marker
        pm = page_marker_re.match(line)
        if pm:
            current_page = int(pm.group(1))
            body_lines.append(line)  # keep in body_lines but filter in _flush
            continue

        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            _flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            node: Dict[str, Any] = {
                "title": title,
                "_level": level,
                "text": "",
                "start_index": current_page,
                "end_index": current_page,
                "nodes": [],
            }
            while stack and stack[-1]["_level"] >= level:
                stack.pop()
            if stack:
                stack[-1]["nodes"].append(node)
            else:
                roots.append(node)
            stack.append(node)
            current = node
        else:
            body_lines.append(line)

    _flush()

    # Propagate end_index up: a parent's end_index should be the max
    # of its own end_index and all its children's end_index values
    def _propagate_end(nodes):
        for n in nodes:
            if n.get("nodes"):
                _propagate_end(n["nodes"])
                child_max = max(c.get("end_index", 0) for c in n["nodes"])
                n["end_index"] = max(n.get("end_index", 0), child_max)

    _propagate_end(roots)

    def _strip(nodes):
        for n in nodes:
            n.pop("_level", None)
            _strip(n.get("nodes", []))

    _strip(roots)

    # Fallback: If no headings were found, return the whole text as one node
    if not roots and text.strip():
        roots = [{
            "title": "General Content",
            "text": text.strip(),
            "start_index": 0,
            "end_index": current_page,
            "nodes": []
        }]

    # Deduplicate repeated paragraphs in all nodes
    _deduplicate_nodes(roots)

    return roots


# ─── Node enrichment ──────────────────────────────────────────────────────

def _number_nodes(nodes: List[Dict], counter: List[int] = None) -> None:
    if counter is None:
        counter = [1]
    for n in nodes:
        n["node_id"] = f"{counter[0]:04d}"
        counter[0] += 1
        if n.get("nodes"):
            _number_nodes(n["nodes"], counter)


def _summarise_nodes(
    nodes: List[Dict], pages: List[str], config: PageIndexConfig
) -> None:
    for n in nodes:
        if not n.get("summary"):
            start = n.get("start_index", 0)
            end   = n.get("end_index", start + 1)
            text  = "\n\n".join(pages[start:end])[:3_000]
            prompt = NODE_SUMMARY.format(
                title=n.get("title", "Untitled"),
                start=start, end=end, text=text,
            )
            try:
                n["summary"] = chat(
                    prompt,
                    model=config.model,
                    api_key=config.api_key,
                    base_url=config.base_url,
                    temperature=0.0,
                    max_tokens=80,
                    quiet=config.quiet,
                    enable_thinking=config.enable_thinking,
            num_ctx=config.num_ctx,
                ).strip()
            except Exception:
                n["summary"] = ""
        if n.get("nodes"):
            _summarise_nodes(n["nodes"], pages, config)


def _embed_text(nodes: List[Dict], pages: List[str]) -> None:
    for n in nodes:
        if not n.get("text"):
            s = n.get("start_index", 0)
            e = n.get("end_index", s + 1)
            # Ensure at least one page is included even when end_index == start_index
            e = max(e, s + 1)
            n["text"] = "\n\n".join(pages[s:e])
        if n.get("nodes"):
            _embed_text(n["nodes"], pages)


def _make_doc_description(nodes: List[Dict], config: PageIndexConfig) -> str:
    preview = json.dumps(
        [{k: v for k, v in n.items() if k not in ("nodes", "text")} for n in nodes[:10]],
        indent=2,
    )
    try:
        return chat(
            DOC_DESCRIPTION.format(tree=preview),
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            temperature=0.0,
            max_tokens=200,
            quiet=config.quiet,
            enable_thinking=config.enable_thinking,
            num_ctx=config.num_ctx,
        ).strip()
    except Exception:
        return ""


# ─── Helpers ──────────────────────────────────────────────────────────────

def _fmt_pages(pages: List[str], offset: int = 0) -> str:
    return "\n\n".join(
        f"[Page {offset + i}]\n{t}" for i, t in enumerate(pages)
    )


def _count_nodes(nodes: List[Dict]) -> int:
    return sum(1 + _count_nodes(n.get("nodes", [])) for n in nodes)


def _fallback_node(start: int, end: int, idx: int) -> Dict:
    return {
        "title": f"Section (pages {start}–{end - 1})",
        "node_id": f"{idx:04d}",
        "start_index": start,
        "end_index": end,
        "summary": "",
        "nodes": [],
    }


def _log(config: PageIndexConfig, msg: str) -> None:
    if config.verbose and not config.quiet:
        print(f"[agenticrag] {msg}")
    log.info(msg)