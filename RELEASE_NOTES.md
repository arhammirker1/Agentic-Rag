# Release v2.1.1: Clone-Free Web UI & SQLite Schema Bugfixes

This release brings highly requested usability upgrades, making the Web UI entirely clone-free, and fixing a critical database migration bug for existing SQLite stores.

## 🚀 Major Feature: Clone-Free Web UI
Previously, `server.py` and the `web/` static files were kept at the repository root and excluded from PyPI packaging. Users were required to clone the GitHub repository to run the Web UI.

In **v2.1.1**, the entire Web UI architecture has been integrated natively into the library package:
- **Zero Cloning**: All UI static assets (HTML/CSS/JS) and the FastAPI server logic are now packaged within the library wheel and tarball.
- **Global Serve Command**: Start the Web UI from anywhere on your machine with a simple one-liner:
  ```bash
  pip install agentic-rag-core[web]
  python -m agenticrag serve
  ```

## 🐛 Bug Fixes
- **SQLite Database Auto-Migration Fix**: Resolved a critical SQLite schema bug (`sqlite3.OperationalError: no such column: parent_doc_id`) that caused crashes on existing databases when upgrading. The `CREATE INDEX` for `parent_doc_id` is now safely deferred until the `_migrate()` auto-schema migration finishes, ensuring older databases are seamlessly upgraded without data loss.

---

# Release v2.1.0: Hybrid Sub-Tree Pre-Filtering

This release introduces a massive architectural optimization for handling large documents (like SEC 10-K filings and extensive manuals): **Hybrid Sub-Tree Pre-Filtering**.

## 🚀 Major Feature: Hybrid Sub-Tree Filtering
Previously, AgenticRAG passed the entire document tree to the LLM during the `SELECT_NODES` phase. For large documents with hundreds of nodes, this caused extreme token bloat (up to 25,000+ tokens per call), triggering rate limits and causing the LLM to miss relevant nodes ("needle in a haystack" problem).

To solve this, we've introduced a pre-filtering pipeline that prunes the tree *before* the LLM sees it, reducing token usage by up to **98%**.

### 1. New Agent: `KeywordAgent`
We've added a dedicated `KeywordAgent`. Instead of searching the tree blindly, this agent takes the user's question and the document's context to generate highly specific **keyphrases, keywords, and synonyms**. It uses the document's own vocabulary to predict what terms will actually appear in the text.

### 2. Local Python Regex Scoring
The expanded keywords are fed into a fast, local Python regex scorer (`_local_node_search`) that evaluates every node without burning any LLM tokens. It scores nodes based on:
- **Stem Title Bonus (+10 pts)**: A smart depluralisation heuristic ensures that plural/singular mismatches (e.g., querying "executives" vs a node titled "Executive Officers") don't cause relevant nodes to be dropped.
- **Deep Text Horizon**: It scans the first 2,000 characters of each node to find keyword hits, ensuring that terms hidden deep inside large tables (like financial statements) are detected.

### 3. Compact Candidate Sub-Trees
The system takes the highest-scoring nodes and rebuilds a "compact sub-tree" that preserves the parent-child relationships. Instead of sending an 800-node tree to the LLM, the Hunter agent now receives a highly targeted 15-node tree, retaining the hierarchical context but eliminating the noise.

## 🛠️ Command to Update
Users can upgrade to the latest version by running:
```bash
pip install agentic-rag-core[web] --upgrade
```
