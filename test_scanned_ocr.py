#!/usr/bin/env python3
"""
test_scanned_ocr.py  --  TEST SCRIPT (not production)

Approach: instead of OCR-to-raw-text, we use ocrmypdf to "bake" an
invisible text layer into the scanned PDF, producing a proper searchable
PDF. pdfplumber then reads it like a native digital PDF -- full layout,
font analysis, tables, etc.

Pipeline:
    scanned.pdf  ->  [ocrmypdf]  ->  searchable.pdf  ->  [pdfplumber]  ->  .md

Requirements (already installed):
    pip install ocrmypdf pymupdf pytesseract pdfplumber
    Tesseract-OCR installed at C:\\Program Files\\Tesseract-OCR\\

Usage:
    python test_scanned_ocr.py sv600_c_automatic.pdf
    python test_scanned_ocr.py sv600_c_automatic.pdf --output-dir ./testresults
    python test_scanned_ocr.py sv600_c_automatic.pdf --keep-searchable
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ── Tesseract path (auto-detected) ────────────────────────────────────────────
TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR",
    r"C:\Program Files (x86)\Tesseract-OCR",
]


def _find_tesseract_dir() -> str | None:
    for p in TESSERACT_PATHS:
        if os.path.exists(os.path.join(p, "tesseract.exe")):
            return p
    return None


# ── Step 1: make the scanned PDF searchable ───────────────────────────────────

def make_searchable_pdf(
    input_pdf: Path,
    output_pdf: Path,
    language: str = "eng",
    deskew: bool = True,
) -> Path:
    """
    Run ocrmypdf to embed a text layer into a scanned PDF.
    Returns output_pdf path.
    """
    tess_dir = _find_tesseract_dir()
    env = os.environ.copy()
    if tess_dir:
        env["TESSDATA_PREFIX"] = os.path.join(tess_dir, "tessdata")
        # Prepend tesseract dir to PATH so ocrmypdf can find it
        env["PATH"] = tess_dir + os.pathsep + env.get("PATH", "")

    cmd = [
        sys.executable, "-m", "ocrmypdf",
        "--language", language,
        "--output-type", "pdf",
        "--optimize", "0",        # don't re-compress images (faster)
        "--jobs", str(os.cpu_count() or 4),  # parallel pages
        "--skip-text",            # skip pages that already have text
    ]
    if deskew:
        cmd.append("--deskew")    # straighten skewed scans

    cmd += [str(input_pdf), str(output_pdf)]

    print(f"  [OCR] Making searchable PDF via ocrmypdf …", flush=True)
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    if result.returncode not in (0, 6):   # 0=success, 6=already has text
        print(f"  [WARN] ocrmypdf stderr:\n{result.stderr}", file=sys.stderr)
        if result.returncode != 0:
            raise RuntimeError(
                f"ocrmypdf failed (exit {result.returncode}):\n{result.stderr}"
            )

    print(f"  [OK]  Searchable PDF -> {output_pdf.name}", flush=True)
    return output_pdf


# ── Step 2: convert searchable PDF → Markdown (reuse existing converter) ──────

def searchable_pdf_to_markdown(pdf_path: Path) -> str:
    """
    Use the existing PDFToMarkdown converter (pdfplumber-based).
    The searchable PDF now has real text, so all the font/layout
    analysis works exactly as for a native digital PDF.
    """
    # Import the main converter from the same directory
    sys.path.insert(0, str(Path(__file__).parent))
    from pdf_to_markdown import PDFToMarkdown
    return PDFToMarkdown(pdf_path).convert()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def convert_scanned_pdf(
    input_pdf: Path,
    output_dir: Path | None = None,
    keep_searchable: bool = False,
    language: str = "eng",
    deskew: bool = True,
) -> Path:
    """
    Full pipeline: scanned PDF → searchable PDF → Markdown.
    Returns the path to the written .md file.
    """
    out_dir = output_dir or input_pdf.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # We'll write the searchable PDF next to the output, or in a temp file
    if keep_searchable:
        searchable_path = out_dir / f"{input_pdf.stem}_searchable.pdf"
        make_searchable_pdf(input_pdf, searchable_path, language, deskew)
        md = searchable_pdf_to_markdown(searchable_path)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            searchable_path = Path(tmp) / f"{input_pdf.stem}_searchable.pdf"
            make_searchable_pdf(input_pdf, searchable_path, language, deskew)
            md = searchable_pdf_to_markdown(searchable_path)
            # temp dir cleaned up automatically

    md_path = out_dir / f"{input_pdf.stem}.md"
    md_path.write_text(md, encoding="utf-8")
    return md_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert scanned PDF → searchable PDF → Markdown (test script)"
    )
    parser.add_argument("pdf", help="Path to the scanned PDF")
    parser.add_argument("--output-dir", default=None, help="Directory for .md output")
    parser.add_argument(
        "--keep-searchable", action="store_true",
        help="Also save the intermediate searchable PDF"
    )
    parser.add_argument(
        "--lang", default="eng",
        help="Tesseract language code(s), e.g. eng, jpn, eng+jpn (default: eng)"
    )
    parser.add_argument(
        "--no-deskew", action="store_true",
        help="Skip deskew correction (faster, but skewed scans stay skewed)"
    )

    args = parser.parse_args()
    pdf_path = Path(args.pdf)

    if not pdf_path.exists():
        print(f"ERROR: File not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else None

    print(f"[*] Input:    {pdf_path.name}")

    try:
        md_path = convert_scanned_pdf(
            pdf_path,
            output_dir=out_dir,
            keep_searchable=args.keep_searchable,
            language=args.lang,
            deskew=not args.no_deskew,
        )
        md = md_path.read_text(encoding="utf-8")
        print(f"[OK] Saved -> {md_path}")
        print(f"     Lines: {len(md.splitlines())} | Size: {len(md):,} bytes")

    except RuntimeError as e:
        print(f"\n[FAIL] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
