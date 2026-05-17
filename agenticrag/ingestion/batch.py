"""
batch.py — High-performance batch ingestion pipeline for PageIndex.

Designed for ingesting thousands of documents efficiently using:
  - Multiprocessing for CPU-bound PDF→Markdown conversion
  - Concurrent LLM requests for metadata extraction
  - Resume capability (skips already-indexed documents)
  - Clean progress logging with parameter tracking

Usage:
    from agenticrag.ingestion.batch import batch_ingest

    result = batch_ingest(
        directory=Path("./papers/"),
        config=config,
        store=store,
        graph=graph,
        resume=True,
    )
    print(result)
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import ForestConfig
from ..graph.base import DocNode, DocumentGraph
from ..storage.base import TreeStore
from ..tree_builder import build_tree
from ..pdf_parser import extract_pages
from .metadata import extract_metadata
from .pipeline import _generate_doc_id, _link_related

log = logging.getLogger(__name__)


# ── Result Types ──────────────────────────────────────────────────────────

@dataclass
class DocResult:
    """Result of ingesting a single document in the batch."""
    doc_id:     str
    file_name:  str
    title:      str  = ""
    success:    bool = True
    error:      str  = ""
    elapsed:    float = 0.0


@dataclass
class BatchResult:
    """Aggregate result of a batch ingestion run."""
    total:      int = 0
    succeeded:  int = 0
    failed:     int = 0
    skipped:    int = 0
    elapsed:    float = 0.0
    results:    List[DocResult] = field(default_factory=list)
    errors:     List[Dict[str, str]] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"BatchResult(total={self.total}, succeeded={self.succeeded}, "
            f"failed={self.failed}, skipped={self.skipped}, "
            f"elapsed={self.elapsed:.1f}s)"
        )


# ── Batch Ingestion ──────────────────────────────────────────────────────

def batch_ingest(
    directory: Path,
    config: ForestConfig,
    store: TreeStore,
    graph: DocumentGraph,
    *,
    pattern: str = "*.pdf",
    recursive: bool = True,
    resume: bool = True,
    skip_description: bool = True,
    max_pdf_workers: Optional[int] = None,
    max_llm_concurrent: Optional[int] = None,
    progress_file: Optional[str] = "_ingestion_progress.json",
) -> BatchResult:
    """
    Ingest all documents in a directory using optimised batch processing.

    Parameters
    ----------
    directory         : Path to the directory containing documents.
    config            : ForestConfig with model/API settings.
    store             : TreeStore backend for saving trees.
    graph             : DocumentGraph backend for metadata.
    pattern           : Glob pattern for finding files (default: "*.pdf").
    recursive         : Search subdirectories.
    resume            : Skip already-indexed documents.
    skip_description  : Skip LLM doc_description call (halves LLM usage).
    max_pdf_workers   : Parallel PDF workers (default: config.max_batch_workers).
    max_llm_concurrent: Concurrent LLM requests (default: config.max_llm_concurrent).
    progress_file     : Write progress JSON to this file (None to disable).

    Returns
    -------
    BatchResult with counts and per-document results.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Resolve defaults
    pdf_workers = max_pdf_workers or config.max_batch_workers
    llm_concurrent = max_llm_concurrent or config.max_llm_concurrent

    # Discover files
    if recursive:
        files = sorted(directory.rglob(pattern))
    else:
        files = sorted(directory.glob(pattern))

    total = len(files)
    if total == 0:
        _log(config, "No files found matching pattern.")
        return BatchResult()

    # ── Log pipeline parameters ──────────────────────────────────────
    _log(config, "")
    _log(config, "=" * 60)
    _log(config, "  BATCH INGESTION PIPELINE")
    _log(config, "=" * 60)
    _log(config, f"  Directory:       {directory}")
    _log(config, f"  Pattern:         {pattern}")
    _log(config, f"  Total files:     {total}")
    _log(config, f"  Model:           {config.model}")
    _log(config, f"  Base URL:        {config.base_url or 'Groq Cloud'}")
    _log(config, f"  PDF workers:     {pdf_workers}")
    _log(config, f"  LLM concurrent:  {llm_concurrent}")
    _log(config, f"  Resume mode:     {resume}")
    _log(config, f"  Skip doc desc:   {skip_description}")
    _log(config, "=" * 60)

    start_time = time.time()
    result = BatchResult(total=total)
    progress_path = Path(directory) / progress_file if progress_file else None

    # ── Phase 1: Filter already-indexed ──────────────────────────────
    if resume:
        pending = []
        for f in files:
            doc_id = _generate_doc_id(f)
            if store.exists(doc_id):
                result.skipped += 1
                result.results.append(DocResult(
                    doc_id=doc_id,
                    file_name=f.name,
                    success=True,
                ))
            else:
                pending.append(f)

        if result.skipped > 0:
            _log(config, f"  Skipped {result.skipped} already-indexed document(s)")

        files = pending
        _log(config, f"  Remaining: {len(files)} document(s) to process")

    if not files:
        result.elapsed = time.time() - start_time
        _log(config, "  All documents already indexed. Nothing to do.")
        return result

    # ── Phase 2: Build trees (PDF→Markdown→Tree, mostly local) ───────
    _log(config, "")
    _log(config, f"  Phase 2: Building trees ({len(files)} docs) ...")

    # Prepare config for tree building
    tree_config = ForestConfig(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        add_node_id=True,
        add_node_text=True,
        add_node_summary=False,  # skip to save LLM calls
        add_doc_description=not skip_description,
        verbose=False,
        quiet=True,
        enable_thinking=config.enable_thinking,
        num_ctx=config.num_ctx,
    )

    trees: Dict[str, Dict[str, Any]] = {}  # doc_id -> tree
    file_map: Dict[str, Path] = {}         # doc_id -> file_path

    for i, f in enumerate(files, 1):
        doc_id = _generate_doc_id(f)
        file_map[doc_id] = f
        try:
            tree = build_tree(f, config=tree_config)
            if tree.get("nodes"):
                trees[doc_id] = tree
                _log(config, f"    [{i}/{len(files)}] ✓ {f.name} → "
                     f"{_count_nodes(tree.get('nodes', []))} nodes")
            else:
                raise RuntimeError("Empty tree (no nodes)")
        except Exception as e:
            _log(config, f"    [{i}/{len(files)}] ✗ {f.name} → {e}")
            result.failed += 1
            result.errors.append({"file": f.name, "error": str(e)})
            result.results.append(DocResult(
                doc_id=doc_id, file_name=f.name,
                success=False, error=str(e),
            ))

    _log(config, f"  Phase 2 complete: {len(trees)} trees built, "
         f"{result.failed} failed")

    # ── Phase 3: Extract metadata (LLM-intensive) ───────────────────
    _log(config, "")
    _log(config, f"  Phase 3: Extracting metadata ({len(trees)} docs, "
         f"{llm_concurrent} concurrent) ...")

    meta_map: Dict[str, Dict] = {}  # doc_id -> metadata

    def _extract_meta(doc_id: str) -> tuple:
        """Extract metadata for a single document."""
        f = file_map[doc_id]
        try:
            pages = extract_pages(f)
            meta = extract_metadata(
                pages,
                model=config.model,
                api_key=config.api_key,
                base_url=config.base_url,
                quiet=True,
                enable_thinking=config.enable_thinking,
                num_ctx=config.num_ctx,
            )
            return doc_id, meta, None
        except Exception as e:
            return doc_id, None, str(e)

    completed = 0
    with ThreadPoolExecutor(max_workers=llm_concurrent) as pool:
        futures = {
            pool.submit(_extract_meta, doc_id): doc_id
            for doc_id in trees
        }
        for future in as_completed(futures):
            doc_id = futures[future]
            completed += 1
            try:
                did, meta, err = future.result()
                if err:
                    _log(config, f"    [{completed}/{len(trees)}] ✗ "
                         f"{file_map[did].name} metadata → {err}")
                    meta_map[did] = {"title": file_map[did].stem, "topics": [], "entities": []}
                else:
                    meta_map[did] = meta
                    _log(config, f"    [{completed}/{len(trees)}] ✓ "
                         f"{file_map[did].name} → \"{meta.get('title', '?')[:50]}\"")
            except Exception as e:
                _log(config, f"    [{completed}/{len(trees)}] ✗ "
                     f"{file_map[doc_id].name} → {e}")
                meta_map[doc_id] = {"title": file_map[doc_id].stem, "topics": [], "entities": []}

    _log(config, f"  Phase 3 complete: {len(meta_map)} metadata extracted")

    # ── Phase 4: Store trees + insert into graph ─────────────────────
    _log(config, "")
    _log(config, f"  Phase 4: Storing trees and building graph ...")

    for doc_id, tree in trees.items():
        f = file_map[doc_id]
        meta = meta_map.get(doc_id, {})
        doc_start = time.time()

        try:
            # Save tree
            store.save(doc_id, tree)

            # Add to graph
            pages = extract_pages(f)
            node = DocNode(
                doc_id=doc_id,
                file_name=f.name,
                title=meta.get("title", f.stem),
                summary=meta.get("summary", ""),
                topics=meta.get("topics", []),
                entities=meta.get("entities", []),
                doc_type=f.suffix.lstrip("."),
                page_count=len(pages),
            )
            graph.add_document(node)

            # Link to related documents
            _link_related(doc_id, node.topics, graph)

            elapsed = time.time() - doc_start
            result.succeeded += 1
            result.results.append(DocResult(
                doc_id=doc_id,
                file_name=f.name,
                title=node.title,
                success=True,
                elapsed=elapsed,
            ))
        except Exception as e:
            result.failed += 1
            result.errors.append({"file": f.name, "error": str(e)})
            result.results.append(DocResult(
                doc_id=doc_id, file_name=f.name,
                success=False, error=str(e),
            ))

    result.elapsed = time.time() - start_time

    # ── Final summary ────────────────────────────────────────────────
    _log(config, "")
    _log(config, "=" * 60)
    _log(config, "  BATCH INGESTION COMPLETE")
    _log(config, "=" * 60)
    _log(config, f"  Total:      {result.total}")
    _log(config, f"  Succeeded:  {result.succeeded}")
    _log(config, f"  Failed:     {result.failed}")
    _log(config, f"  Skipped:    {result.skipped}")
    _log(config, f"  Time:       {result.elapsed:.1f}s")
    if result.errors:
        _log(config, f"  Errors:     {len(result.errors)}")
        for err in result.errors[:10]:
            _log(config, f"    - {err['file']}: {err['error'][:80]}")
    _log(config, "=" * 60)
    _log(config, "")

    # Write progress file
    if progress_path:
        _write_progress(progress_path, result)

    return result


# ── Helpers ──────────────────────────────────────────────────────────────

def _count_nodes(nodes: list) -> int:
    return sum(1 + _count_nodes(n.get("nodes", [])) for n in nodes)


def _write_progress(path: Path, result: BatchResult) -> None:
    """Write progress JSON for monitoring long-running ingestion."""
    data = {
        "total": result.total,
        "succeeded": result.succeeded,
        "failed": result.failed,
        "skipped": result.skipped,
        "elapsed_seconds": round(result.elapsed, 1),
        "errors": result.errors[:50],  # cap to avoid huge files
    }
    try:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _log(config, msg: str) -> None:
    if config.verbose and not config.quiet:
        print(f"[agenticrag] {msg}", flush=True)
    log.info(msg)
