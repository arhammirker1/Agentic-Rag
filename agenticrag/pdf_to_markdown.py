#!/usr/bin/env python3
"""
pdf_to_markdown.py  --  High-quality PDF -> Markdown converter (no LLM).

Uses pdfplumber character-level font metadata (size, bold, position) to detect
headings, sub-headings, tables, bullet lists, and figure captions.
Produces clean, structured Markdown suitable for RAG ingestion.

Works on any PDF -- no document-specific heuristics.

Usage:
    python pdf_to_markdown.py "report.pdf"
    python pdf_to_markdown.py report.pdf -o report.md
    python pdf_to_markdown.py report.pdf --output-dir ./results
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class TextLine:
    """A single line of text with its font metadata."""
    text: str
    y_top: float
    y_bottom: float
    x_left: float
    x_right: float
    font_size: float
    is_bold: bool
    is_italic: bool
    page_num: int
    page_width: float = 612.0
    in_table: bool = False

    @property
    def height(self) -> float:
        return max(self.y_bottom - self.y_top, 1.0)

    @property
    def is_short(self) -> bool:
        """Line occupies less than 55% of page width."""
        return (self.x_right - self.x_left) < self.page_width * 0.55

    @property
    def words(self) -> int:
        return len(self.text.split())


@dataclass
class FontProfile:
    body_size: float = 0.0
    all_sizes: List[float] = field(default_factory=list)


@dataclass
class Table:
    page: int
    y_top: float
    y_bottom: float
    markdown: str
    used: bool = False


# ============================================================================
# Converter
# ============================================================================

class PDFToMarkdown:
    """
    Pure-heuristic PDF -> Markdown converter.

    Core logic:
      1. Scan all characters to find the body font size (most frequent).
      2. Extract tables via pdfplumber table detection.
      3. Group remaining characters into lines with font metadata.
      4. Classify each line: heading / bullet / body / figure caption.
      5. Merge continuation lines into paragraphs and bullet points.
      6. Interleave tables at the correct position.
    """

    BULLET_RE = re.compile(
        r'^(?:'
        r'[\u2022\u2023\u25CF\u25CB\u25A0\u25AA\u25B8\u2013\u2014\u25B6\u25BA]\s+'  # unicode bullets
        r'|\d{1,3}[\.\)]\s+'       # "1. " or "1) "
        r'|[a-zA-Z][\.\)]\s+'      # "a. " or "a) "
        r'|\([a-zA-Z0-9]+\)\s+'    # "(1) " or "(a) "
        r'|[-*+]\s+'               # markdown-style bullets
        r')'
    )

    BULLET_CHARS = frozenset(
        '\u2022\u2023\u25CF\u25CB\u25A0\u25AA\u25B8\u2013\u2014\u25B6\u25BA'
    )

    FIGURE_PREFIXES = (
        'figure:', 'figure ', 'fig.', 'fig ',
        'chart:', 'chart ', 'table:', 'table ',
        'diagram:', 'diagram ',
    )

    def __init__(self, pdf_path: str | Path):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("pdfplumber required:  pip install pdfplumber")

        self.pdf = pdfplumber.open(str(self.pdf_path))
        self.profile = FontProfile()
        self.lines: List[TextLine] = []
        self.tables: List[Table] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def convert(self) -> str:
        self._profile_fonts()
        self._extract_tables()
        self._extract_lines()
        md = self._build_markdown()
        md = self._cleanup(md)
        self.pdf.close()
        return md

    # ------------------------------------------------------------------
    # 1. Font profiling
    # ------------------------------------------------------------------

    def _profile_fonts(self):
        ctr: Counter = Counter()
        for page in self.pdf.pages:
            for ch in page.chars:
                s = round(float(ch.get("size", 0)), 1)
                if s > 0:
                    ctr[s] += 1
        if not ctr:
            self.profile.body_size = 10.0
            return
        self.profile.body_size = ctr.most_common(1)[0][0]
        self.profile.all_sizes = sorted(ctr.keys(), reverse=True)

    # ------------------------------------------------------------------
    # 2. Table extraction
    # ------------------------------------------------------------------

    def _extract_tables(self):
        for pi, page in enumerate(self.pdf.pages):
            for tobj in page.find_tables():
                rows = tobj.extract()
                if not rows:
                    continue
                md = self._table_to_md(rows)
                if md:
                    bbox = tobj.bbox
                    self.tables.append(Table(
                        page=pi, y_top=bbox[1], y_bottom=bbox[3], markdown=md,
                    ))

    @staticmethod
    def _table_to_md(rows: List[List]) -> str:
        """Convert extracted table rows into a markdown table."""
        if not rows:
            return ""

        # Clean each cell
        clean: List[List[str]] = []
        for row in rows:
            clean.append([
                re.sub(r'\s*\n\s*', ' ', str(c or '').strip()) for c in row
            ])

        # Normalize column count
        ncols = max(len(r) for r in clean)
        for r in clean:
            while len(r) < ncols:
                r.append('')

        # Skip fully empty tables
        if all(all(c == '' for c in r) for r in clean):
            return ""

        # Skip if header is all empty
        if all(c == '' for c in clean[0]) and len(clean) > 1:
            clean = clean[1:]

        # Column widths for alignment
        widths = [max(3, *(len(r[i]) for r in clean)) for i in range(ncols)]

        def fmt(cells):
            return '| ' + ' | '.join(
                cells[i].ljust(widths[i]) for i in range(ncols)
            ) + ' |'

        out = [fmt(clean[0])]
        out.append('| ' + ' | '.join('-' * w for w in widths) + ' |')
        for r in clean[1:]:
            out.append(fmt(r))
        return '\n'.join(out)

    # ------------------------------------------------------------------
    # 3. Line extraction
    # ------------------------------------------------------------------

    def _extract_lines(self):
        for pi, page in enumerate(self.pdf.pages):
            if not page.chars:
                continue
            pw = float(page.width) if page.width else 612.0
            pt = [t for t in self.tables if t.page == pi]
            self.lines.extend(self._group_chars(page.chars, pi, pt, pw))

    def _group_chars(self, chars, page_num, page_tables, pw) -> List[TextLine]:
        """Group characters into lines by y-position proximity."""
        sc = sorted(chars, key=lambda c: (round(float(c['top']), 1), float(c['x0'])))
        groups: List[List[dict]] = [[sc[0]]]

        for ch in sc[1:]:
            if abs(float(ch['top']) - float(groups[-1][-1]['top'])) < 2.5:
                groups[-1].append(ch)
            else:
                groups.append([ch])

        result = []
        for g in groups:
            tl = self._build_line(g, page_num, page_tables, pw)
            if tl and tl.text.strip():
                result.append(tl)
        return result

    def _build_line(self, chars, page_num, page_tables, pw) -> Optional[TextLine]:
        """Build a TextLine from a group of same-y characters."""
        sc = sorted(chars, key=lambda c: float(c['x0']))

        # Reconstruct text with spacing
        parts: List[str] = []
        px1 = None
        for ch in sc:
            x0 = float(ch['x0'])
            if px1 is not None and (x0 - px1) > max(float(ch.get('width', 5)), 1) * 0.3:
                parts.append(' ')
            parts.append(ch.get('text', ''))
            px1 = float(ch['x1'])

        text = ''.join(parts).strip()
        if not text:
            return None

        # Only consider visible (non-whitespace) chars for font analysis
        vis = [c for c in chars if c.get('text', '').strip()]
        if not vis:
            return None

        sizes = [round(float(c.get('size', 0)), 1) for c in vis]
        names = [c.get('fontname', '') for c in vis]

        dom_size = Counter(sizes).most_common(1)[0][0]
        dom_font = Counter(names).most_common(1)[0][0].lower()
        bold = any(k in dom_font for k in ('bold', 'heavy', 'black', 'demi'))
        italic = any(k in dom_font for k in ('italic', 'oblique', 'slant'))

        yt = min(float(c['top']) for c in chars)
        yb = max(float(c['bottom']) for c in chars)
        xl = min(float(c['x0']) for c in chars)
        xr = max(float(c['x1']) for c in chars)

        in_table = any(yt >= t.y_top - 2 and yb <= t.y_bottom + 2 for t in page_tables)

        return TextLine(
            text=text, y_top=yt, y_bottom=yb, x_left=xl, x_right=xr,
            font_size=dom_size, is_bold=bold, is_italic=italic,
            page_num=page_num, page_width=pw, in_table=in_table,
        )

    # ------------------------------------------------------------------
    # 4. Line classification
    # ------------------------------------------------------------------

    def _classify(self, line: TextLine) -> str:
        """
        Classify a line as one of:
          title, h1, h2, h3, bullet, figure_cap, body, skip
        """
        text = line.text.strip()
        bs = self.profile.body_size

        if not text or line.in_table:
            return 'skip'

        # Form-feed / page-break artifacts
        if text in ('\f', '\x0c'):
            return 'skip'

        # Figure / chart captions
        if any(text.lower().startswith(p) for p in self.FIGURE_PREFIXES) and len(text) > 8:
            return 'figure_cap'

        # ---- Size-based heading detection ----
        diff = line.font_size - bs

        if diff >= 6:
            return 'title'       # e.g. 22pt vs 10pt body
        if diff >= 3:
            return 'h1'          # e.g. 14pt
        if diff >= 1.5:
            return 'h2'          # e.g. 12pt
        if diff >= 0.5 and line.is_bold:
            return 'h3'          # slightly larger + bold

        # Bold-only headings: bold, short, not a sentence-ending line
        if (line.is_bold and line.font_size >= bs
                and line.is_short and line.words <= 12
                and not text.endswith(('.', ',', ';'))):
            return 'h3'

        # Bullets
        if self.BULLET_RE.match(text) or (text[0] in self.BULLET_CHARS):
            return 'bullet'

        return 'body'

    # ------------------------------------------------------------------
    # 5. Markdown assembly
    # ------------------------------------------------------------------

    def _build_markdown(self) -> str:
        out: List[str] = []
        used_tables: set = set()
        i, n = 0, len(self.lines)
        current_page = -1  # Track page changes

        def _emit_tables(page: int, y: float):
            """Emit any tables positioned before (page, y)."""
            for ti, tb in enumerate(self.tables):
                if ti in used_tables:
                    continue
                if tb.page < page or (tb.page == page and tb.y_bottom <= y + 5):
                    out.append('')
                    out.append(tb.markdown)
                    out.append('')
                    used_tables.add(ti)

        def _emit_page_marker(page: int):
            """Emit a page marker comment when page changes."""
            nonlocal current_page
            if page != current_page:
                out.append(f'<!-- page:{page} -->')
                current_page = page

        while i < n:
            line = self.lines[i]
            role = self._classify(line)

            if role == 'skip':
                i += 1
                continue

            _emit_page_marker(line.page_num)
            _emit_tables(line.page_num, line.y_top)

            # ---- Headings ----
            if role in ('title', 'h1', 'h2', 'h3'):
                prefix = {'title': '#', 'h1': '##', 'h2': '###', 'h3': '####'}[role]
                out.append('')
                out.append(f'{prefix} {line.text.strip()}')
                out.append('')
                i += 1

            # ---- Figure captions ----
            elif role == 'figure_cap':
                out.append('')
                out.append(f'> *{line.text.strip()}*')
                out.append('')
                i += 1

            # ---- Bullets (with continuation merging) ----
            elif role == 'bullet':
                bt = line.text.strip()
                # Normalize bullet char to nothing (we add "- " ourselves)
                bt = re.sub(
                    r'^[\u2022\u2023\u25CF\u25CB\u25A0\u25AA\u25B8'
                    r'\u2013\u2014\u25B6\u25BA]\s*', '', bt,
                )
                last = line
                i += 1
                # Merge continuation body lines
                while i < n:
                    nxt = self.lines[i]
                    nr = self._classify(nxt)
                    if nr == 'skip':
                        i += 1
                        continue
                    if nr == 'body':
                        gap = nxt.y_top - last.y_bottom
                        same_or_next = (nxt.page_num == last.page_num
                                        or nxt.page_num == last.page_num + 1)
                        if same_or_next and gap < last.height * 2.0:
                            bt += ' ' + nxt.text.strip()
                            last = nxt
                            i += 1
                            continue
                    break
                if bt:
                    out.append(f'- {bt}')

            # ---- Body paragraphs (with continuation merging) ----
            elif role == 'body':
                parts = [line.text.strip()]
                prev = line
                i += 1
                while i < n:
                    nxt = self.lines[i]
                    nr = self._classify(nxt)
                    if nr == 'skip':
                        i += 1
                        continue
                    if nr != 'body':
                        break
                    gap = nxt.y_top - prev.y_bottom
                    # Paragraph break: big gap on the same page
                    if nxt.page_num == prev.page_num and gap > prev.height * 1.8:
                        break
                    # Track page changes within merged paragraphs
                    _emit_page_marker(nxt.page_num)
                    parts.append(nxt.text.strip())
                    prev = nxt
                    i += 1

                out.append('')
                out.append(' '.join(parts))
                out.append('')

            else:
                i += 1

        # Emit remaining tables
        for ti, tb in enumerate(self.tables):
            if ti not in used_tables:
                out.append('')
                out.append(tb.markdown)
                out.append('')

        return '\n'.join(out)

    # ------------------------------------------------------------------
    # 6. Final cleanup
    # ------------------------------------------------------------------

    @staticmethod
    def _cleanup(md: str) -> str:
        # Collapse 3+ blank lines to 2
        md = re.sub(r'\n{4,}', '\n\n\n', md)

        # Strip trailing whitespace per line
        md = '\n'.join(l.rstrip() for l in md.split('\n'))

        # Ensure blank line before first bullet in a list
        md = re.sub(r'([^\n])\n(- )', r'\1\n\n\2', md)

        # Ensure blank line after last bullet before non-bullet content
        md = re.sub(r'(- [^\n]+)\n([^-\n])', r'\1\n\n\2', md)

        # Remove duplicate consecutive headings (same text)
        lines = md.split('\n')
        result = []
        prev_heading = None
        for line in lines:
            if line.startswith('#'):
                key = line.strip().lower()
                if key == prev_heading:
                    continue
                prev_heading = key
            else:
                prev_heading = None
            result.append(line)

        return '\n'.join(result).strip() + '\n'


# ============================================================================
# Public API
# ============================================================================

def convert_pdf_to_markdown(pdf_path: str | Path) -> str:
    """
    Convert a PDF file to well-structured Markdown.

    Uses font-size analysis, bold detection, table extraction, and
    bullet-point recognition. No LLM calls -- fast and free.
    """
    return PDFToMarkdown(pdf_path).convert()


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Convert PDF to structured Markdown (no LLM required)'
    )
    parser.add_argument('pdf', help='Path to the PDF file')
    parser.add_argument('-o', '--output', default=None, help='Output .md path')
    parser.add_argument('--output-dir', default=None, help='Directory for output')
    parser.add_argument('--stdout', action='store_true', help='Print to stdout')

    args = parser.parse_args()
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f'ERROR: File not found: {pdf_path}', file=sys.stderr)
        sys.exit(1)

    print(f'[*] Converting: {pdf_path.name}')
    markdown = convert_pdf_to_markdown(pdf_path)

    if args.stdout:
        print(markdown)
        return

    if args.output:
        out_path = Path(args.output)
    elif args.output_dir:
        od = Path(args.output_dir)
        od.mkdir(parents=True, exist_ok=True)
        out_path = od / f'{pdf_path.stem}.md'
    else:
        out_path = pdf_path.with_suffix('.md')

    out_path.write_text(markdown, encoding='utf-8')
    print(f'[OK] Saved -> {out_path}')
    print(f'     Lines: {len(markdown.splitlines())} | Size: {len(markdown):,} bytes')


if __name__ == '__main__':
    main()
