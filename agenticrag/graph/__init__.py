"""
agenticrag.graph — Pluggable document graph backends.

The document graph stores metadata about every indexed document
(title, topics, entities, summary) and the relationships between them.
The Planner Agent queries this graph to find which documents are
relevant to a user's question.

Available backends:
  - SQLiteGraph  : SQLite (default, zero dependencies, persistent)
  - Neo4jGraph   : Neo4j (production scale, requires `pip install agenticrag[neo4j]`)
"""

from .base import DocumentGraph, DocNode
from .sqlite_graph import SQLiteGraph

__all__ = ["DocumentGraph", "DocNode", "SQLiteGraph"]

try:
    from .neo4j_graph import Neo4jGraph
    __all__.append("Neo4jGraph")
except ImportError:
    pass
