# Release v2.1.2: CWD-Aware Dotenv Auto-Loading

This patch release fixes a critical issue where the library failed to discover and load API keys (e.g., `GROQ_API_KEY`, `GEMINI_API_KEY`) from the user's project `.env` file when the package was installed from PyPI.

## 🛠️ Bugfix: CWD-Aware Dotenv Auto-Loading
Previously, `load_dotenv()` was called without arguments inside the packaged library's `__init__.py` and `server.py`. In `python-dotenv`, this defaults to searching for `.env` starting from the library's installation folder (deep within `site-packages`) and moving upwards, which completely missed the user's active project directory (CWD).

We have updated the dotenv initialization across all entry points:
- **CWD Resolution**: Instructed dotenv to start searching starting from the active process's current working directory using `find_dotenv(usecwd=True)`.
- **Force Override**: Enabled `override=True` so that empty/blank environment variables (commonly set by IDEs or container environments) are properly overridden by values defined in the `.env` file.

## 🛠️ Command to Update
Users can upgrade to this version by running:
```bash
pip install agentic-rag-core[web] --upgrade
```

---

# Release v2.1.1: Hybrid Sub-Tree Pre-Filtering & Clone-Free Web UI

This release introduces a massive architectural optimization for handling large documents (like SEC 10-K filings and extensive manuals): **Hybrid Sub-Tree Pre-Filtering**, alongside a completely rewritten **Clone-Free Web UI** distribution and a database schema migration bugfix.

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

## 🎨 Clone-Free Web UI & SQLite Migration Fix
You can now run the premium web interface directly from the PyPI library without needing to clone the GitHub repository! We have restructured and packaged all the server and frontend assets right inside the core library.

- **Internal Packaging**: The FastAPI `server.py` and the complete static `web/` assets are now fully bundled in the python wheel package.
- **SQLite Schema Auto-Migration Fix**: Solved the `sqlite3.OperationalError: no such column: parent_doc_id` database upgrade issue. SQLite indexes are now cleanly generated *after* column auto-migration runs, ensuring users with older version databases upgrade automatically and painlessly.
- **Easy CLI Command**: Spin up the Web UI from any directory on your computer:
  ```bash
  pip install agentic-rag-core[web] --upgrade
  python -m agenticrag serve
  ```

## 🛠️ Command to Update
Users can upgrade to this version by running:
```bash
pip install agentic-rag-core[web] --upgrade
```
