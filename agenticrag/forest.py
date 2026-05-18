"""
forest.py — Multi-document knowledge Forest.

This is the top-level entry point for multi-document reasoning-based RAG.
It manages a collection of indexed documents and answers questions across
them using a multi-agent pipeline.

Quick start:
    from agenticrag import Forest

    forest = Forest(api_key="gsk_...")
    forest.add("report.pdf")
    forest.add("contract.pdf")
    forest.add_directory("./documents/")

    result = forest.ask("What are the key financial risks?")
    print(result.text)           # The synthesized answer
    print(result.sources)        # Which documents/pages were used
    print(result.confidence)     # How confident the system is
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .agents.orchestrator import ForestResult, Orchestrator
from .config import ForestConfig, GroqModel, PageIndexConfig
from .graph.base import DocumentGraph
from .graph.sqlite_graph import SQLiteGraph
from .ingestion.pipeline import IngestResult, ingest_document
from .storage.base import TreeStore
from .storage.local import LocalStore

log = logging.getLogger(__name__)


class Forest:
    """
    Multi-document knowledge base with agentic retrieval.

    The Forest manages:
    - A **TreeStore** for persisting document tree indices (JSON files).
    - A **DocumentGraph** for metadata and inter-document relationships.
    - An **Orchestrator** that runs the multi-agent pipeline on queries.

    Parameters
    ----------
    api_key : str or None
        Your Groq API key.  Falls back to GROQ_API_KEY env var.
    model : str
        Groq model ID.  Default: openai/gpt-oss-20b (fast, good quality).
    data_dir : str or Path
        Root directory for all persisted data (trees, graph DB).
        Default: "./pageindex_data"
    store : TreeStore or None
        Custom tree storage backend.  If None, uses LocalStore(data_dir/trees).
    graph : DocumentGraph or None
        Custom graph backend.  If None, uses SQLiteGraph(data_dir/graph.db).
    verbose : bool
        Print progress to stdout.
    **config_kwargs
        Additional PageIndexConfig options.

    Example
    -------
    >>> forest = Forest(api_key="gsk_...")
    >>> forest.add("annual_report.pdf")
    >>> forest.add("quarterly_update.pdf")
    >>> result = forest.ask("What was the revenue growth?")
    >>> print(result.text)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = GroqModel.GPT_OSS_20B,
        data_dir: str | Path = "./pageindex_data",
        store: Optional[TreeStore] = None,
        graph: Optional[DocumentGraph] = None,
        verbose: bool = False,
        **config_kwargs: Any,
    ):
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        # Build configuration
        self.config = ForestConfig(
            model=model,
            api_key=api_key,
            verbose=verbose,
            data_dir=str(data_dir),
            **config_kwargs,
        )

        # Initialise storage backends
        self.store: TreeStore = store or LocalStore(data_dir / "trees")
        self.graph: DocumentGraph = graph or SQLiteGraph(data_dir / "graph.db")

        # Initialise orchestrator
        self._orchestrator = Orchestrator(
            config=self.config,
            store=self.store,
            graph=self.graph,
        )

        self._history: List[Dict[str, str]] = []

    # ── Document Management ──────────────────────────────────────────────

    def add(
        self,
        path: Union[str, Path],
        doc_id: Optional[str] = None,
    ) -> IngestResult:
        """
        Add a document to the Forest.

        Indexes the document, extracts metadata, and stores everything
        persistently.  Subsequent calls with the same file are no-ops
        (the document is already indexed).

        Parameters
        ----------
        path   : Path to the document (.pdf, .md, .txt).
        doc_id : Optional custom identifier.  Auto-generated if None.

        Returns
        -------
        IngestResult with doc_id, title, topics, and success status.
        """
        return ingest_document(
            file_path=path,
            config=self.config,
            store=self.store,
            graph=self.graph,
            doc_id=doc_id,
        )

    def add_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "*.pdf",
        recursive: bool = True,
    ) -> List[IngestResult]:
        """
        Add all matching documents from a directory (sequential).

        For large collections (100+ docs), use ``add_directory_batch()`` instead.

        Parameters
        ----------
        directory : Path to the directory.
        pattern   : Glob pattern (default: "*.pdf").
                    Use "**/*" patterns for mixed types.
        recursive : If True, search subdirectories.

        Returns
        -------
        List of IngestResult, one per document.
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        if recursive:
            files = list(directory.rglob(pattern))
        else:
            files = list(directory.glob(pattern))

        results = []
        total = len(files)
        for i, f in enumerate(files, 1):
            if self.config.verbose:
                print(f"[agenticrag] [{i}/{total}] Processing {f.name} …")
            result = self.add(f)
            results.append(result)

        successes = sum(1 for r in results if r.success)
        if self.config.verbose:
            print(f"[agenticrag] [OK] Indexed {successes}/{total} documents.")

        return results

    def add_directory_batch(
        self,
        directory: Union[str, Path],
        pattern: str = "*.pdf",
        recursive: bool = True,
        resume: bool = True,
        skip_description: bool = True,
        max_pdf_workers: Optional[int] = None,
        max_llm_concurrent: Optional[int] = None,
    ):
        """
        Add documents from a directory using the high-performance batch pipeline.

        Optimised for large collections (100-100K+ documents) with:
          - Concurrent LLM requests for metadata extraction
          - Resume capability (skips already-indexed docs)
          - Progress tracking with clean logging

        Parameters
        ----------
        directory          : Path to the directory.
        pattern            : Glob pattern (default: "*.pdf").
        recursive          : Search subdirectories.
        resume             : Skip already-indexed documents (default: True).
        skip_description   : Skip doc_description LLM call to halve ingestion
                             time (default: True). The metadata summary serves
                             the same purpose for the Planner.
        max_pdf_workers    : Parallel PDF→Markdown workers (default: config setting).
        max_llm_concurrent : Concurrent LLM requests (default: config setting).

        Returns
        -------
        BatchResult with total/succeeded/failed/skipped counts and per-doc results.

        Example
        -------
        >>> result = forest.add_directory_batch("./papers/", resume=True)
        >>> print(result)  # BatchResult(total=14000, succeeded=13950, ...)
        """
        from .ingestion.batch import batch_ingest

        return batch_ingest(
            directory=Path(directory),
            config=self.config,
            store=self.store,
            graph=self.graph,
            pattern=pattern,
            recursive=recursive,
            resume=resume,
            skip_description=skip_description,
            max_pdf_workers=max_pdf_workers,
            max_llm_concurrent=max_llm_concurrent,
        )

    def remove(self, doc_id: str) -> None:
        """
        Remove a document from the Forest.

        Deletes the tree from storage and removes the node from the graph.
        """
        self.store.delete(doc_id)
        self.graph.remove_document(doc_id)

    def documents(self, include_parts: bool = False) -> List[Dict[str, Any]]:
        """
        List all indexed documents with their metadata.

        Parameters
        ----------
        include_parts : If False (default), hide sub-tree parts and show
                        only top-level documents.  Set True to see all
                        sub-trees.

        Returns
        -------
        List of dicts with doc_id, title, topics, summary, etc.
        """
        nodes = self.graph.list_documents()
        result = []
        for n in nodes:
            # Skip sub-tree parts unless requested
            if not include_parts and n.parent_doc_id is not None:
                continue
            result.append({
                "doc_id": n.doc_id,
                "file_name": n.file_name,
                "title": n.title,
                "summary": n.summary,
                "topics": n.topics,
                "page_count": n.page_count,
            })
        return result

    @property
    def size(self) -> int:
        """Number of documents in the Forest."""
        return self.graph.count()

    # ── Querying ─────────────────────────────────────────────────────────

    def ask(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        skip_critic: bool = False,
    ) -> ForestResult:
        """
        Ask a question across all indexed documents.

        Uses the full multi-agent pipeline:
          Planner → Hunters (parallel) → Synthesizer → Critic

        Parameters
        ----------
        question    : Your question (free text).
        history     : Optional conversation history for multi-turn.
        skip_critic : Skip the hallucination check (faster but less safe).

        Returns
        -------
        ForestResult with:
          .text               — The verified answer
          .sources            — Which documents/pages were used
          .confidence         — 0.0 to 1.0 confidence score
          .documents_searched — Which doc_ids were searched
          .reasoning_trace    — Step-by-step agent trace
          .was_rewritten      — Whether the Critic modified the answer
          .elapsed_seconds    — Total time taken

        Example
        -------
        >>> result = forest.ask("What are the key risks?")
        >>> print(result.text)
        >>> for source in result.sources:
        ...     print(f"  {source['doc_title']} — Pages {source['pages']}")
        """
        # Use provided history or internal history
        effective_history = history if history is not None else self._history

        result = self._orchestrator.ask(
            question,
            history=effective_history,
            skip_critic=skip_critic,
        )

        # Update internal history for multi-turn
        if history is None:
            self._history.append({"role": "user", "content": question})
            self._history.append({"role": "assistant", "content": result.text})
            # Keep last 10 turns to avoid context overflow
            if len(self._history) > 20:
                self._history = self._history[-20:]

        return result

    def clear_history(self) -> None:
        """Clear the internal conversation history."""
        self._history.clear()

    # ── Utilities ────────────────────────────────────────────────────────

    def info(self) -> Dict[str, Any]:
        """Return a summary of the Forest state."""
        return {
            "documents": self.size,
            "data_dir": self.config.data_dir,
            "model": self.config.model,
            "store": repr(self.store),
            "graph": repr(self.graph),
        }

    def __repr__(self) -> str:
        return f"Forest(documents={self.size}, model='{self.config.model}')"

    def __len__(self) -> int:
        return self.size
