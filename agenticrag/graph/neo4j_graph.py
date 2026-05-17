"""
neo4j_graph.py — Neo4j-backed document knowledge graph.

For production workloads with millions of documents and complex
relationship queries.

Requires:  pip install neo4j
       or: pip install pageindex[neo4j]

Usage:
    graph = Neo4jGraph(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="your_password",
    )
    graph.add_document(DocNode(doc_id="abc", file_name="report.pdf", ...))
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import DocNode, DocumentGraph

try:
    from neo4j import GraphDatabase  # type: ignore
except ImportError:
    raise ImportError(
        "Neo4j support requires the `neo4j` package.\n"
        "Install it with:  pip install neo4j\n"
        "Or:               pip install pageindex[neo4j]"
    )


class Neo4jGraph(DocumentGraph):
    """
    Neo4j-backed document graph for production-scale deployments.

    Parameters
    ----------
    uri      : Bolt URI (e.g. "bolt://localhost:7687")
    user     : Neo4j username
    password : Neo4j password
    database : Database name (default: "neo4j")
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "",
        database: str = "neo4j",
    ):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._db = database
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        with self._driver.session(database=self._db) as session:
            session.run(
                "CREATE INDEX doc_id_idx IF NOT EXISTS FOR (d:Document) ON (d.doc_id)"
            )
            session.run(
                "CREATE FULLTEXT INDEX doc_search_idx IF NOT EXISTS "
                "FOR (d:Document) ON EACH [d.title, d.summary, d.topics_str, d.entities_str]"
            )

    # ── Node operations ──────────────────────────────────────────────────

    def add_document(self, node: DocNode) -> None:
        with self._driver.session(database=self._db) as session:
            session.run(
                """
                MERGE (d:Document {doc_id: $doc_id})
                SET d.file_name   = $file_name,
                    d.title       = $title,
                    d.summary     = $summary,
                    d.topics      = $topics,
                    d.topics_str  = $topics_str,
                    d.entities    = $entities,
                    d.entities_str = $entities_str,
                    d.doc_type    = $doc_type,
                    d.page_count  = $page_count
                """,
                doc_id=node.doc_id,
                file_name=node.file_name,
                title=node.title,
                summary=node.summary,
                topics=node.topics,
                topics_str=" ".join(node.topics),
                entities=node.entities,
                entities_str=" ".join(node.entities),
                doc_type=node.doc_type,
                page_count=node.page_count,
            )

    def get_document(self, doc_id: str) -> Optional[DocNode]:
        with self._driver.session(database=self._db) as session:
            result = session.run(
                "MATCH (d:Document {doc_id: $doc_id}) RETURN d",
                doc_id=doc_id,
            )
            record = result.single()
            return self._record_to_node(record["d"]) if record else None

    def remove_document(self, doc_id: str) -> None:
        with self._driver.session(database=self._db) as session:
            session.run(
                "MATCH (d:Document {doc_id: $doc_id}) DETACH DELETE d",
                doc_id=doc_id,
            )

    def list_documents(self) -> List[DocNode]:
        with self._driver.session(database=self._db) as session:
            result = session.run("MATCH (d:Document) RETURN d")
            return [self._record_to_node(r["d"]) for r in result]

    def count(self) -> int:
        with self._driver.session(database=self._db) as session:
            result = session.run("MATCH (d:Document) RETURN count(d) AS c")
            return result.single()["c"]

    # ── Edge operations ──────────────────────────────────────────────────

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        rel_type: str = "RELATED",
        weight: float = 1.0,
    ) -> None:
        # Neo4j relationship types must be uppercase with no spaces
        safe_type = rel_type.upper().replace(" ", "_")
        with self._driver.session(database=self._db) as session:
            session.run(
                f"""
                MATCH (a:Document {{doc_id: $source}})
                MATCH (b:Document {{doc_id: $target}})
                MERGE (a)-[r:{safe_type}]->(b)
                SET r.weight = $weight
                """,
                source=source_id,
                target=target_id,
                weight=weight,
            )

    def get_neighbors(
        self,
        doc_id: str,
        rel_type: Optional[str] = None,
    ) -> List[DocNode]:
        with self._driver.session(database=self._db) as session:
            if rel_type:
                safe_type = rel_type.upper().replace(" ", "_")
                result = session.run(
                    f"""
                    MATCH (a:Document {{doc_id: $doc_id}})-[:{safe_type}]->(b:Document)
                    RETURN b
                    """,
                    doc_id=doc_id,
                )
            else:
                result = session.run(
                    """
                    MATCH (a:Document {doc_id: $doc_id})-->(b:Document)
                    RETURN b
                    """,
                    doc_id=doc_id,
                )
            return [self._record_to_node(r["b"]) for r in result]

    # ── Search ───────────────────────────────────────────────────────────

    def search_by_topics(
        self,
        topics: List[str],
        limit: int = 10,
    ) -> List[DocNode]:
        if not topics:
            return []
        with self._driver.session(database=self._db) as session:
            result = session.run(
                """
                MATCH (d:Document)
                WITH d, [t IN $topics WHERE t IN d.topics] AS matches
                WHERE size(matches) > 0
                RETURN d, size(matches) AS score
                ORDER BY score DESC
                LIMIT $limit
                """,
                topics=topics,
                limit=limit,
            )
            return [self._record_to_node(r["d"]) for r in result]

    def search_by_text(
        self,
        query: str,
        limit: int = 10,
    ) -> List[DocNode]:
        with self._driver.session(database=self._db) as session:
            try:
                result = session.run(
                    """
                    CALL db.index.fulltext.queryNodes('doc_search_idx', $query)
                    YIELD node, score
                    RETURN node AS d
                    ORDER BY score DESC
                    LIMIT $limit
                    """,
                    query=query,
                    limit=limit,
                )
                return [self._record_to_node(r["d"]) for r in result]
            except Exception:
                # Fallback: CONTAINS search
                result = session.run(
                    """
                    MATCH (d:Document)
                    WHERE d.title CONTAINS $query OR d.summary CONTAINS $query
                    RETURN d
                    LIMIT $limit
                    """,
                    query=query,
                    limit=limit,
                )
                return [self._record_to_node(r["d"]) for r in result]

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _record_to_node(record) -> DocNode:
        props = dict(record)
        return DocNode(
            doc_id=props.get("doc_id", ""),
            file_name=props.get("file_name", ""),
            title=props.get("title", ""),
            summary=props.get("summary", ""),
            topics=list(props.get("topics", [])),
            entities=list(props.get("entities", [])),
            doc_type=props.get("doc_type", "pdf"),
            page_count=props.get("page_count", 0),
        )

    def close(self) -> None:
        self._driver.close()

    def __repr__(self) -> str:
        return f"Neo4jGraph(documents={self.count()})"

    def __del__(self):
        try:
            self._driver.close()
        except Exception:
            pass
