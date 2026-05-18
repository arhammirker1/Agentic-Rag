"""
agenticrag — Vectorless, Reasoning-based RAG
=============================================

Build hierarchical tree indices from any document (PDF, Markdown, TXT),
then use LLM reasoning (via Groq) to retrieve the most relevant sections
for any question — no vector DB, no chunking, no embeddings.

Single document:
    from agenticrag import PageIndex

    pi = PageIndex(api_key="gsk_...")
    pi.load("report.pdf")
    answer = pi.ask("What was the net income?")
    print(answer.text)

Multi-document (Forest):
    from agenticrag import Forest

    forest = Forest(api_key="gsk_...")
    forest.add("report_2023.pdf")
    forest.add("report_2024.pdf")
    forest.add_directory("./contracts/")

    result = forest.ask("Compare revenue between 2023 and 2024")
    print(result.text)
    print(result.sources)
    print(result.confidence)
"""

import os
from dotenv import load_dotenv, find_dotenv

# Automatically load .env file so API keys are discovered without extra code.
# Search starting from the current working directory (CWD) of the running process,
# and override any existing (possibly blank) env vars.
load_dotenv(find_dotenv(usecwd=True), override=True)

# ── Core (single document) ───────────────────────────────────────────────
from .config import PageIndexConfig, ForestConfig, GroqModel, LocalModel
from .tree_builder import build_tree
from .tree_search import TreeSearcher, SearchResult
from .pdf_parser import extract_pages
from .pageindex import PageIndex

# ── Forest (multi-document) ──────────────────────────────────────────────
from .forest import Forest
from .agents.orchestrator import ForestResult

# ── Storage & Graph (for advanced users) ─────────────────────────────────
from .storage import TreeStore, LocalStore
from .graph import DocumentGraph, DocNode, SQLiteGraph

__all__ = [
    # High-level APIs
    "PageIndex",             # single-document
    "Forest",                # multi-document (recommended)
    "ForestResult",          # result type for Forest.ask()
    # Configuration
    "PageIndexConfig",
    "ForestConfig",
    "GroqModel",             # cloud model IDs (Groq)
    "LocalModel",            # local model IDs (Ollama)
    # Core building blocks
    "build_tree",
    "TreeSearcher",
    "SearchResult",
    "extract_pages",
    # Storage backends
    "TreeStore",
    "LocalStore",
    # Graph backends
    "DocumentGraph",
    "DocNode",
    "SQLiteGraph",
]

__version__ = "2.1.3"
__author__  = "Arham Mirkar"
__license__ = "MIT"