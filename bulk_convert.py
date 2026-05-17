#!/usr/bin/env python3
"""
bulk_convert.py  --  Parallel bulk PDF -> Markdown converter.

Converts an entire folder of PDFs (including scanned ones) to Markdown
using all available CPU cores.

Strategy:
  - Each PDF is processed in its own worker process (true parallelism)
  - Scanned PDFs are auto-detected and OCR'd via ocrmypdf Python API
  - Already-converted files are skipped (resume-friendly)
  - Non-scanned PDFs skip OCR entirely (very fast)

Realistic throughput:
  - Digital PDFs:  ~100-500 docs/min per core
  - Scanned PDFs:  ~4-8 docs/min per core (OCR-bound)
  - With 8 cores:  ~32-64 scanned docs/min  ->  14k docs in ~3-7 hours
  - With 16 cores: ~64-128 scanned docs/min ->  14k docs in ~1.5-4 hours

For true 5-10 min on 14k scanned docs you need GPU OCR or cloud.
See --help for GPU / cloud options info.

Usage:
    python bulk_convert.py ./pdfs_folder --output-dir ./markdowns
    python bulk_convert.py ./pdfs_folder --output-dir ./markdowns --workers 16
    python bulk_convert.py ./pdfs_folder --output-dir ./markdowns --workers 16 --no-deskew
    python bulk_convert.py ./pdfs_folder --output-dir ./markdowns --resume
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# ── Worker function (runs in a separate process) ───────────────────────────────

def _worker_convert(args_tuple) -> dict:
    """
    Convert a single PDF to Markdown.
    Must be a module-level function for pickling by multiprocessing.
    Returns a result dict with status info.
    """
    pdf_path, out_path, ocr_lang, deskew = args_tuple
    t0 = time.perf_counter()

    try:
        # Import here so each worker process initialises its own state
        sys.path.insert(0, str(Path(__file__).parent))
        from pdf_to_markdown import PDFToMarkdown

        md = PDFToMarkdown(pdf_path, ocr_lang=ocr_lang, deskew=deskew).convert()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")

        elapsed = time.perf_counter() - t0
        return {
            "status": "ok",
            "pdf": str(pdf_path),
            "out": str(out_path),
            "lines": len(md.splitlines()),
            "size": len(md),
            "elapsed": elapsed,
        }

    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {
            "status": "error",
            "pdf": str(pdf_path),
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed": elapsed,
        }


# ── Progress display ───────────────────────────────────────────────────────────

class ProgressBar:
    def __init__(self, total: int):
        self.total = total
        self.done = 0
        self.ok = 0
        self.errors = 0
        self.skipped = 0
        self.t_start = time.perf_counter()

    def update(self, status: str):
        self.done += 1
        if status == "ok":
            self.ok += 1
        elif status == "error":
            self.errors += 1
        elif status == "skip":
            self.skipped += 1
        self._print()

    def _print(self):
        elapsed = time.perf_counter() - self.t_start
        rate = self.done / elapsed if elapsed > 0 else 0
        remaining = (self.total - self.done) / rate if rate > 0 else 0
        pct = self.done / self.total * 100

        bar_w = 30
        filled = int(bar_w * self.done / self.total)
        bar = "█" * filled + "░" * (bar_w - filled)

        eta_str = _fmt_time(remaining)
        elapsed_str = _fmt_time(elapsed)

        print(
            f"\r[{bar}] {pct:5.1f}%  "
            f"{self.done}/{self.total}  "
            f"ok:{self.ok} err:{self.errors} skip:{self.skipped}  "
            f"{rate:.1f} doc/s  "
            f"elapsed {elapsed_str}  ETA {eta_str}   ",
            end="", flush=True
        )

    def finish(self):
        print()   # newline after the bar


def _fmt_time(seconds: float) -> str:
    seconds = max(0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# ── Main bulk converter ────────────────────────────────────────────────────────

def bulk_convert(
    input_dir: Path,
    output_dir: Path,
    workers: int,
    ocr_lang: str = "eng",
    deskew: bool = True,
    resume: bool = True,
    recursive: bool = True,
) -> dict:
    """
    Convert all PDFs in input_dir to Markdown files in output_dir.
    Returns a summary dict.
    """
    # Collect all PDFs
    glob = "**/*.pdf" if recursive else "*.pdf"
    all_pdfs = sorted(input_dir.glob(glob))

    if not all_pdfs:
        print(f"No PDFs found in: {input_dir}")
        return {}

    print(f"\n[*] Found {len(all_pdfs):,} PDF(s) in {input_dir}")
    print(f"    Output dir : {output_dir}")
    print(f"    Workers    : {workers}")
    print(f"    Language   : {ocr_lang}")
    print(f"    Deskew     : {deskew}")
    print(f"    Resume     : {resume}\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build work queue, skipping already-done files if --resume
    tasks = []
    skipped_upfront = 0
    for pdf in all_pdfs:
        # Mirror the input directory structure in the output
        rel = pdf.relative_to(input_dir)
        out_md = output_dir / rel.with_suffix(".md")

        if resume and out_md.exists() and out_md.stat().st_size > 10:
            skipped_upfront += 1
            continue

        tasks.append((str(pdf), out_md, ocr_lang, deskew))

    total_to_process = len(tasks)
    print(f"[*] To convert : {total_to_process:,}  |  Already done (skipped): {skipped_upfront:,}\n")

    if total_to_process == 0:
        print("[OK] Nothing to do. Use --no-resume to reconvert everything.")
        return {"total": len(all_pdfs), "skipped": skipped_upfront, "ok": 0, "errors": 0}

    # Estimate time
    est_secs = total_to_process * 10 / workers   # rough: 10s/doc single-threaded
    print(f"[~] Rough estimate: {_fmt_time(est_secs)}  (assumes ~10s/doc, actual varies)")
    print(f"    (Digital PDFs are much faster; scanned PDFs are OCR-bound)\n")

    bar = ProgressBar(total_to_process)
    errors: List[dict] = []
    t_start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_to_task = {
            pool.submit(_worker_convert, task): task
            for task in tasks
        }

        for future in as_completed(future_to_task):
            result = future.result()
            bar.update(result["status"])
            if result["status"] == "error":
                errors.append(result)

    bar.finish()
    total_elapsed = time.perf_counter() - t_start

    # ── Summary ────────────────────────────────────────────────────────────────
    ok_count = bar.ok
    err_count = bar.errors
    rate = total_to_process / total_elapsed if total_elapsed > 0 else 0

    print(f"\n{'='*60}")
    print(f"  DONE in {_fmt_time(total_elapsed)}")
    print(f"  Converted : {ok_count:,}")
    print(f"  Errors    : {err_count:,}")
    print(f"  Skipped   : {skipped_upfront:,}")
    print(f"  Throughput: {rate:.2f} docs/sec  ({rate*60:.1f} docs/min)")
    print(f"{'='*60}\n")

    if errors:
        err_log = output_dir / "_errors.txt"
        with err_log.open("w", encoding="utf-8") as f:
            for e in errors:
                f.write(f"{e['pdf']}\n  {e['error']}\n\n")
        print(f"  Error log -> {err_log}")

    return {
        "total": len(all_pdfs),
        "processed": total_to_process,
        "ok": ok_count,
        "errors": err_count,
        "skipped": skipped_upfront,
        "elapsed_sec": total_elapsed,
        "rate_per_sec": rate,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    cpu_count = os.cpu_count() or 4
    default_workers = max(1, cpu_count - 1)   # leave 1 core free for OS

    parser = argparse.ArgumentParser(
        description="Bulk-convert a folder of PDFs (including scanned) to Markdown in parallel.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
SPEED TIPS:
  --workers 16        Use more cores (check os.cpu_count())
  --no-deskew         Skip deskew (saves ~30%% time on well-scanned docs)
  --resume            Skip already-converted files (great for re-runs)
  --lang eng          Only process the language you need

WANT FASTER? (beyond CPU limits)
  GPU OCR:    pip install easyocr  (requires NVIDIA GPU + CUDA)
              -- can be 5-10x faster than Tesseract on GPU
  Cloud OCR:  Google Cloud Vision / Azure Computer Vision / AWS Textract
              -- can parallelize across hundreds of machines
              -- 14k docs in ~5-10 min is realistic at cloud scale
        """
    )
    parser.add_argument("input_dir", help="Folder containing PDFs")
    parser.add_argument("--output-dir", required=True, help="Where to write .md files")
    parser.add_argument(
        "--workers", type=int, default=default_workers,
        help=f"Parallel worker processes (default: {default_workers}, your CPU has {cpu_count} cores)"
    )
    parser.add_argument(
        "--lang", default="eng",
        help="Tesseract language for scanned PDFs (default: eng). Use eng+jpn for mixed."
    )
    parser.add_argument(
        "--no-deskew", action="store_true",
        help="Skip deskew (faster, worse for tilted scans)"
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Reconvert even files that already have a .md output"
    )
    parser.add_argument(
        "--flat", action="store_true",
        help="Don't recurse into subdirectories"
    )

    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        print(f"ERROR: Not a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)

    bulk_convert(
        input_dir=input_dir,
        output_dir=output_dir,
        workers=args.workers,
        ocr_lang=args.lang,
        deskew=not args.no_deskew,
        resume=not args.no_resume,
        recursive=not args.flat,
    )


if __name__ == "__main__":
    main()
