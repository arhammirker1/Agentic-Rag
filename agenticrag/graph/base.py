"""
base.py — Abstract base class for the document knowledge graph.

Every graph backend must implement this interface.  The graph stores
document metadata and inter-document relationships so the Planner Agent
can quickly identify which documents are relevant to a question.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DocNode:
    """
    Metadata for a single document in the knowledge graph.

    Attributes
    ----------
    doc_id     : Unique identifier (usually a hash of the file).
    file_name  : Original filename.
    title      : Document title (extracted by LLM).
    summary    : 2-3 sentence summary of the document.
    topics     : List of key topics / themes.
    entities   : List of named entities (people, orgs, products, etc.).
    doc_type   : Document type (pdf, markdown, txt).
    page_count : Number of pages.
    extra      : Any additional metadata (flexible dict).
    """
    doc_id:     str
    file_name:  str
    title:      str             = ""
    summary:    str             = ""
    topics:     List[str]       = field(default_factory=list)
    entities:   List[str]       = field(default_factory=list)
    doc_type:   str             = "pdf"
    page_count: int             = 0
    parent_doc_id: Optional[str] = None   # set for sub-trees of a split document
    extra:      Dict[str, Any]  = field(default_factory=dict)


class DocumentGraph(ABC):
    """
    Abstract interface for the document knowledge graph.

    The graph has two core primitives:
      - Nodes  : documents (DocNode)
      - Edges  : relationships between documents (shared topics, citations, etc.)
    """

    # ── Node operations ──────────────────────────────────────────────────

    @abstractmethod
    def add_document(self, node: DocNode) -> None:
        """Insert or update a document node in the graph."""
        ...

    @abstractmethod
    def get_document(self, doc_id: str) -> Optional[DocNode]:
        """Return the DocNode for doc_id, or None."""
        ...

    @abstractmethod
    def remove_document(self, doc_id: str) -> None:
        """Remove a document and all its edges from the graph."""
        ...

    @abstractmethod
    def list_documents(self) -> List[DocNode]:
        """Return all document nodes."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Return the total number of documents in the graph."""
        ...

    # ── Edge operations ──────────────────────────────────────────────────

    @abstractmethod
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        rel_type: str = "related",
        weight: float = 1.0,
    ) -> None:
        """Create a directed edge between two documents."""
        ...

    @abstractmethod
    def get_neighbors(
        self,
        doc_id: str,
        rel_type: Optional[str] = None,
    ) -> List[DocNode]:
        """Return documents connected to doc_id, optionally filtered by rel_type."""
        ...

    # ── Search / query ───────────────────────────────────────────────────

    @abstractmethod
    def search_by_topics(
        self,
        topics: List[str],
        limit: int = 10,
    ) -> List[DocNode]:
        """
        Find documents whose topics overlap with the given list.
        Returns results ranked by relevance (most topic overlap first).
        """
        ...

    @abstractmethod
    def search_by_text(
        self,
        query: str,
        limit: int = 10,
    ) -> List[DocNode]:
        """
        Full-text search over document titles, summaries, topics, and entities.
        Returns results ranked by relevance.
        """
        ...
