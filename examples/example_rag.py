"""
example_rag.py — End-to-end AgenticRAG demo

This shows the complete vectorless RAG pipeline:
  1. Extract pages from a PDF
  2. Build the tree index
  3. Ask questions with context-aware retrieval
"""

import json
import os
from pathlib import Path

from agenticrag import (
    PageIndexConfig,
    TreeSearcher,
    build_tree,
    extract_pages,
)

# ── Configuration ──────────────────────────────────────────────────────────
PDF_PATH = "your_document.pdf"  # ← change this

config = PageIndexConfig(
    model="openai/gpt-oss-20b",
    toc_check_pages=20,
    max_pages_per_node=10,
    add_node_summary=True,
    add_doc_description=True,
    verbose=True,
)

# ── Step 1: Build the tree ─────────────────────────────────────────────────
print("Building tree index ...")
pages = extract_pages(PDF_PATH)
tree = build_tree(PDF_PATH, config)

# Inspect the tree
print(json.dumps(tree, indent=2)[:2000], "…")

# Save to disk
with open("my_document_tree.json", "w") as f:
    json.dump(tree, f, indent=2)

# ── Step 2: Q&A with reasoning-based retrieval ────────────────────────────
print("\nStarting Q&A ...\n")
searcher = TreeSearcher(tree, config=config, pages=pages)

questions = [
    "What is the main topic of this document?",
    "What are the key findings or conclusions?",
    "Are there any risks or limitations mentioned?",
]

history = []
for q in questions:
    print(f"Q: {q}")
    result = searcher.answer(q, history=history)
    print(f"A: {result.text}")
    print(f"   Nodes read: {[n.get('node_id') for n in result.retrieved_nodes]}")
    print(f"   Iterations: {result.iterations}\n")

    # Maintain multi-turn context
    history.append({"role": "user", "content": q})
    history.append({"role": "assistant", "content": result.text})
