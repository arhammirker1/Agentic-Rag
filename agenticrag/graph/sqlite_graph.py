"""
sqlite_graph.py — SQLite-backed document knowledge graph.

This is the default backend.  Uses Python's built-in sqlite3 module,
so it requires ZERO external dependencies and persists to a single file.

For production workloads with millions of documents, consider Neo4jGraph.

Usage:
    graph = SQLiteGraph("./data/graph.db")
    graph.add_document(DocNode(doc_id="abc", file_name="report.pdf", ...))
    results = graph.search_by_topics(["revenue", "Q4"])
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import DocNode, DocumentGraph


class SQLiteGraph(DocumentGraph):
    """
    SQLite-backed document graph.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database file.
        Created automatically if it doesn't exist.
    """

    def __init__(self, db_path: str | Path = "./pageindex_data/graph.db"):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id          TEXT PRIMARY KEY,
                file_name       TEXT NOT NULL,
                title           TEXT DEFAULT '',
                summary         TEXT DEFAULT '',
                topics          TEXT DEFAULT '[]',
                entities        TEXT DEFAULT '[]',
                doc_type        TEXT DEFAULT 'pdf',
                page_count      INTEGER DEFAULT 0,
                parent_doc_id   TEXT DEFAULT NULL,
                extra           TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS edges (
                source_id   TEXT NOT NULL,
                target_id   TEXT NOT NULL,
                rel_type    TEXT DEFAULT 'related',
                weight      REAL DEFAULT 1.0,
                PRIMARY KEY (source_id, target_id, rel_type),
                FOREIGN KEY (source_id) REFERENCES documents(doc_id) ON DELETE CASCADE,
                FOREIGN KEY (target_id) REFERENCES documents(doc_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_topics ON documents(topics);
            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
            CREATE INDEX IF NOT EXISTS idx_parent ON documents(parent_doc_id);
        """)
        # Enable FTS5 for full-text search (built into SQLite)
        self._conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                doc_id, title, summary, topics, entities,
                content=documents,
                content_rowid=rowid
            )
        """)
        self._conn.commit()

        # Auto-migration: add parent_doc_id column if upgrading from older schema
        self._migrate()

    def _rebuild_fts(self) -> None:
        """Rebuild the FTS index after inserts/updates."""
        self._conn.execute("INSERT INTO documents_fts(documents_fts) VALUES('rebuild')")
        self._conn.commit()

    def _migrate(self) -> None:
        """Add columns that may be missing in older databases."""
        cursor = self._conn.execute("PRAGMA table_info(documents)")
        columns = {row[1] for row in cursor.fetchall()}
        if "parent_doc_id" not in columns:
            self._conn.execute(
                "ALTER TABLE documents ADD COLUMN parent_doc_id TEXT DEFAULT NULL"
            )
            self._conn.commit()

    # ── Node operations ──────────────────────────────────────────────────

    def add_document(self, node: DocNode) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO documents
                (doc_id, file_name, title, summary, topics, entities,
                 doc_type, page_count, parent_doc_id, extra)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                node.doc_id,
                node.file_name,
                node.title,
                node.summary,
                json.dumps(node.topics),
                json.dumps(node.entities),
                node.doc_type,
                node.page_count,
                node.parent_doc_id,
                json.dumps(node.extra),
            ),
        )
        self._conn.commit()
        self._rebuild_fts()

    def get_document(self, doc_id: str) -> Optional[DocNode]:
        row = self._conn.execute(
            "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return self._row_to_node(row) if row else None

    def remove_document(self, doc_id: str) -> None:
        # Cascade: also remove sub-trees (children) of this document
        children = self._conn.execute(
            "SELECT doc_id FROM documents WHERE parent_doc_id = ?", (doc_id,)
        ).fetchall()
        for child in children:
            self.remove_document(child["doc_id"])

        self._conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        self._conn.execute(
            "DELETE FROM edges WHERE source_id = ? OR target_id = ?",
            (doc_id, doc_id),
        )
        self._conn.commit()
        self._rebuild_fts()

    def list_documents(self) -> List[DocNode]:
        rows = self._conn.execute("SELECT * FROM documents").fetchall()
        return [self._row_to_node(r) for r in rows]

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()
        return row[0] if row else 0

    # ── Edge operations ──────────────────────────────────────────────────

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        rel_type: str = "related",
        weight: float = 1.0,
    ) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO edges (source_id, target_id, rel_type, weight)
            VALUES (?, ?, ?, ?)
            """,
            (source_id, target_id, rel_type, weight),
        )
        self._conn.commit()

    def get_neighbors(
        self,
        doc_id: str,
        rel_type: Optional[str] = None,
    ) -> List[DocNode]:
        if rel_type:
            rows = self._conn.execute(
                """
                SELECT d.* FROM documents d
                JOIN edges e ON d.doc_id = e.target_id
                WHERE e.source_id = ? AND e.rel_type = ?
                """,
                (doc_id, rel_type),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT d.* FROM documents d
                JOIN edges e ON d.doc_id = e.target_id
                WHERE e.source_id = ?
                """,
                (doc_id,),
            ).fetchall()
        return [self._row_to_node(r) for r in rows]

    # ── Search ───────────────────────────────────────────────────────────

    def search_by_topics(
        self,
        topics: List[str],
        limit: int = 10,
    ) -> List[DocNode]:
        """
        Find documents with overlapping topics.
        Ranks by number of matching topics (descending).
        """
        if not topics:
            return []

        # Use LIKE queries for topic matching (works across all SQLite versions)
        conditions = " OR ".join(["topics LIKE ?"] * len(topics))
        params = [f"%{t}%" for t in topics]
        params.append(str(limit))

        rows = self._conn.execute(
            f"""
            SELECT * FROM documents
            WHERE {conditions}
            LIMIT ?
            """,
            params,
        ).fetchall()

        # Score and sort by number of matching topics
        scored = []
        for row in rows:
            node = self._row_to_node(row)
            score = sum(
                1 for t in topics
                if any(t.lower() in nt.lower() for nt in node.topics)
            )
            scored.append((score, node))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in scored]

    def search_by_text(
        self,
        query: str,
        limit: int = 10,
    ) -> List[DocNode]:
        """Full-text search using SQLite FTS5."""
        try:
            rows = self._conn.execute(
                """
                SELECT d.* FROM documents d
                JOIN documents_fts f ON d.doc_id = f.doc_id
                WHERE documents_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, limit),
            ).fetchall()
            return [self._row_to_node(r) for r in rows]
        except sqlite3.OperationalError:
            # FTS may not be available — fall back to LIKE search
            rows = self._conn.execute(
                """
                SELECT * FROM documents
                WHERE title LIKE ? OR summary LIKE ? OR topics LIKE ? OR entities LIKE ?
                LIMIT ?
                """,
                (f"%{query}%",) * 4 + (limit,),
            ).fetchall()
            return [self._row_to_node(r) for r in rows]

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_node(row: sqlite3.Row) -> DocNode:
        return DocNode(
            doc_id=row["doc_id"],
            file_name=row["file_name"],
            title=row["title"],
            summary=row["summary"],
            topics=json.loads(row["topics"]),
            entities=json.loads(row["entities"]),
            doc_type=row["doc_type"],
            page_count=row["page_count"],
            parent_doc_id=row["parent_doc_id"] if "parent_doc_id" in row.keys() else None,
            extra=json.loads(row["extra"]),
        )

    def get_children(self, parent_doc_id: str) -> List[DocNode]:
        """Return all sub-tree documents that belong to a parent document."""
        rows = self._conn.execute(
            "SELECT * FROM documents WHERE parent_doc_id = ?",
            (parent_doc_id,),
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __repr__(self) -> str:
        return f"SQLiteGraph('{self._db_path}', documents={self.count()})"

    def __del__(self):
        try:
            self._conn.close()
        except Exception:
            pass
