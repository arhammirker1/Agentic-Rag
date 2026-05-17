#!/usr/bin/env python3
"""
run_pageindex.py — CLI for PageIndex (free & open source)

Usage examples:
  # Build a tree from a PDF
  python run_pageindex.py --pdf_path report.pdf

  # Build a tree from a Markdown file
  python run_pageindex.py --md_path notes.md

  # Use a different model (any LiteLLM-compatible string)
  python run_pageindex.py --pdf_path report.pdf --model claude-3-5-sonnet-20241022

  # Interactive Q&A after indexing
  python run_pageindex.py --pdf_path report.pdf --qa
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="PageIndex — Vectorless, Reasoning-based RAG (free & open source)"
    )

    # Input
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdf_path", type=str, help="Path to a PDF file")
    group.add_argument("--md_path", type=str, help="Path to a Markdown file")

    # Config overrides
    parser.add_argument("--model", default="gpt-4o",
                        help="LiteLLM model string (default: gpt-4o)")
    parser.add_argument("--toc-check-pages", type=int, default=20,
                        help="Pages to scan for an existing TOC (PDF only)")
    parser.add_argument("--max-pages-per-node", type=int, default=10,
                        help="Max pages each tree node can span (PDF only)")
    parser.add_argument("--max-tokens-per-node", type=int, default=20000,
                        help="Max tokens per node before splitting")
    parser.add_argument("--no-node-id", action="store_true",
                        help="Omit node IDs from the output")
    parser.add_argument("--no-summary", action="store_true",
                        help="Skip generating node summaries")
    parser.add_argument("--no-description", action="store_true",
                        help="Skip generating the document description")
    parser.add_argument("--add-text", action="store_true",
                        help="Embed raw page text inside each node")
    parser.add_argument("--output-dir", default="./results",
                        help="Directory to save the JSON output")
    parser.add_argument("--verbose", action="store_true",
                        help="Print progress messages")

    # Interactive Q&A
    parser.add_argument("--qa", action="store_true",
                        help="After indexing, enter interactive Q&A mode")

    args = parser.parse_args()

    # Build config
    from agenticrag import PageIndexConfig
    config = PageIndexConfig(
        model=args.model,
        toc_check_pages=args.toc_check_pages,
        max_pages_per_node=args.max_pages_per_node,
        max_tokens_per_node=args.max_tokens_per_node,
        add_node_id=not args.no_node_id,
        add_node_summary=not args.no_summary,
        add_doc_description=not args.no_description,
        add_node_text=args.add_text,
        verbose=args.verbose,
    )

    # Build tree
    from agenticrag import build_tree, extract_pages
    
    path = Path(args.pdf_path or args.md_path)
    if not path.exists():
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)
        
    pages = extract_pages(path)
    tree = build_tree(path, config)
    stem = path.stem

    # Save output
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{stem}_pageindex.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(tree, f, indent=2, ensure_ascii=False)
    print(f"[OK] Tree saved -> {out_file}")

    # Optional doc description
    if tree.get("document_description"):
        print(f"\n[DOC] {tree['document_description']}\n")

    # Interactive Q&A
    if args.qa:
        _interactive_qa(tree, pages, config)


def _interactive_qa(tree, pages, config):
    from agenticrag import TreeSearcher

    searcher = TreeSearcher(tree, config=config, pages=pages)
    history = []
    print("\n[?] PageIndex Q&A  (type 'quit' to exit)\n" + "-" * 40)

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not question:
            continue

        print("[...] Searching ...")
        result = searcher.answer(question, history=history)

        print(f"\nPageIndex: {result.text}\n")
        print(f"  Nodes read: {[n.get('node_id') for n in result.retrieved_nodes]}")
        print(f"  Iterations: {result.iterations}\n")

        # Update history for context-aware retrieval
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": result.text})


if __name__ == "__main__":
    main()
