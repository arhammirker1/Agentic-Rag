"""
pipeline.py — Document ingestion pipeline.

Orchestrates the full flow:
  1. Extract pages from document
  2. Build the PageIndex tree
  3. Extract metadata (title, topics, entities)
  4. Store the tree in the TreeStore
  5. Insert a DocNode into the DocumentGraph
  6. Create edges to related documents (shared topics)
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..config import PageIndexConfig
from ..graph.base import DocNode, DocumentGraph
from ..pdf_parser import extract_pages
from ..storage.base import TreeStore
from ..tree_builder import build_tree
from .metadata import extract_metadata

log = logging.getLogger(__name__)


@dataclass
class IngestResult:
    """
    Result of ingesting a single document.

    Attributes
    ----------
    doc_id     : The unique identifier assigned to this document.
    file_name  : Original filename.
    title      : Extracted document title.
    topics     : Extracted topics.
    page_count : Number of pages.
    success    : Whether ingestion succeeded.
    error      : Error message if failed.
    """
    doc_id:     str
    file_name:  str
    title:      str  = ""
    topics:     list = None
    page_count: int  = 0
    success:    bool = True
    error:      str  = ""

    def __post_init__(self):
        if self.topics is None:
            self.topics = []


def _generate_doc_id(file_path: Path) -> str:
    """Generate a deterministic doc_id from the file path and content hash."""
    content = file_path.read_bytes()
    file_hash = hashlib.sha256(content).hexdigest()[:12]
    stem = file_path.stem.replace(" ", "_")[:40]
    return f"{stem}_{file_hash}"


def ingest_document(
    file_path: str | Path,
    *,
    config: PageIndexConfig,
    store: TreeStore,
    graph: DocumentGraph,
    doc_id: Optional[str] = None,
) -> IngestResult:
    """
    Ingest a single document into the PageIndex Forest.

    This is the core ingestion function.  It:
    1. Extracts pages from the document.
    2. Builds a PageIndex tree.
    3. Extracts metadata (title, topics, entities, summary).
    4. Saves the tree to the TreeStore.
    5. Adds the document to the DocumentGraph.
    6. Links it to existing documents with shared topics.

    Parameters
    ----------
    file_path : Path to the document (.pdf, .md, .txt).
    config    : PageIndexConfig.
    store     : TreeStore backend (where tree JSON is saved).
    graph     : DocumentGraph backend (where metadata is stored).
    doc_id    : Optional custom doc_id. If None, auto-generated.

    Returns
    -------
    IngestResult with doc_id, title, topics, and success status.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return IngestResult(
            doc_id=doc_id or "",
            file_name=file_path.name,
            success=False,
            error=f"File not found: {file_path}",
        )

    # Generate doc_id
    if doc_id is None:
        doc_id = _generate_doc_id(file_path)

    # Check if already indexed
    if store.exists(doc_id):
        log.info(f"Document '{doc_id}' already indexed, skipping.")
        existing = graph.get_document(doc_id)
        return IngestResult(
            doc_id=doc_id,
            file_name=file_path.name,
            title=existing.title if existing else "",
            topics=existing.topics if existing else [],
            page_count=existing.page_count if existing else 0,
            success=True,
        )

    try:
        # 1. Extract pages
        _log(config, f"Extracting pages from {file_path.name} ...")
        pages = extract_pages(file_path)

        # 2. Build tree
        _log(config, f"Building tree index ({len(pages)} pages) ...")
        tree = build_tree(file_path, config=config)
        
        if not tree.get("nodes") and not tree.get("document_description"):
            raise RuntimeError("Tree building failed completely (likely due to LLM rate limits).")

        # 3. Extract metadata
        _log(config, "Extracting metadata ...")
        meta = extract_metadata(pages, model=config.model, api_key=config.api_key, base_url=config.base_url, quiet=config.quiet)

        # 4. Save tree
        store.save(doc_id, tree)

        # 5. Add to graph
        node = DocNode(
            doc_id=doc_id,
            file_name=file_path.name,
            title=meta.get("title", file_path.stem),
            summary=meta.get("summary", ""),
            topics=meta.get("topics", []),
            entities=meta.get("entities", []),
            doc_type=file_path.suffix.lstrip("."),
            page_count=len(pages),
        )
        graph.add_document(node)

        # 6. Link to related documents (shared topics)
        _link_related(doc_id, node.topics, graph)

        _log(config, f"Done: {meta.get('title', file_path.name)}")

        return IngestResult(
            doc_id=doc_id,
            file_name=file_path.name,
            title=node.title,
            topics=node.topics,
            page_count=len(pages),
            success=True,
        )

    except Exception as e:
        log.error(f"Ingestion failed for {file_path}: {e}")
        return IngestResult(
            doc_id=doc_id,
            file_name=file_path.name,
            success=False,
            error=str(e),
        )


def _link_related(
    doc_id: str,
    topics: list,
    graph: DocumentGraph,
    min_shared: int = 2,
) -> None:
    """
    Create edges between the new document and existing documents
    that share at least `min_shared` topics.
    """
    if not topics:
        return

    candidates = graph.search_by_topics(topics, limit=20)
    for candidate in candidates:
        if candidate.doc_id == doc_id:
            continue
        shared = set(t.lower() for t in topics) & set(
            t.lower() for t in candidate.topics
        )
        if len(shared) >= min_shared:
            weight = len(shared) / max(len(topics), 1)
            graph.add_edge(doc_id, candidate.doc_id, "shared_topics", weight)
            graph.add_edge(candidate.doc_id, doc_id, "shared_topics", weight)


def _log(config: PageIndexConfig, msg: str) -> None:
    if config.verbose and not config.quiet:
        print(f"[agenticrag] {msg}")
    log.info(msg)
