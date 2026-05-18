# AgenticRAG: Deep Technical Architecture & Agent Specifications

AgenticRAG is a **vectorless, reasoning-first Retrieval-Augmented Generation (RAG)** engine. Unlike traditional vector-based RAG architectures that convert arbitrary chunks of text into numerical vectors and run mathematical similarity searches (which strip away document structure, hierarchical relationships, and table boundaries), AgenticRAG maintains an explicit, structurally aware model of documents.

By indexing documents as hierarchical **Headings-and-Content trees** (a "Forest") and navigating them using a coordinated state machine of specialized LLM agents, AgenticRAG achieves unmatched accuracy in complex structured text, eliminates multi-step search failure, and guarantees zero-hallucination outputs through self-correcting validation.

---

## 🗺️ End-to-End System Architecture

The lifecycle of a document within AgenticRAG spans two main pipelines: the **Ingestion & Indexing Pipeline** (from user upload to database storage) and the **Agentic Query Pipeline** (from query input to validated synthesized response).

```mermaid
graph TD
    %% Upload Pipeline
    subgraph Ingestion Pipeline (Upload)
        A[User Uploads PDF] --> B[PDF-to-Markdown Parser]
        B --> C[Hierarchical Tree Builder]
        C --> D[Generate Topics & Summary]
        D --> E[Store in SQLite Graph DB]
        E --> F[Register FTS5 Full-Text Index]
    end

    %% Query Pipeline
    subgraph Query Pipeline (Search & Synthesize)
        Q[User Query] --> G[Planner Agent]
        G --> H[Document Selection]
        H --> I[Keyword Agent]
        I --> J[Local Regex Node Scorer]
        J --> K[Compact Sub-Tree Construction]
        K --> L[Hunter Agent]
        L --> M[Evaluator Agent]
        M --> N[Synthesizer Agent]
        N --> O[Critic Agent]
        O -->|Fails Verification| M
        O -->|Passes Verification| P[Final Answer with Citations]
    end

    style Ingestion Pipeline (Upload) fill:#f5f7fa,stroke:#cbd5e1,stroke-width:2px;
    style Query Pipeline (Search & Synthesize) fill:#f0f9ff,stroke:#bae6fd,stroke-width:2px;
```

---

## 📁 1. The Ingestion & Indexing Pipeline (Upload Stage)

When a document is added to the AgenticRAG system (via `Forest.add("file.pdf")` or the Web UI), it goes through three major stages:

### A. Parser: PDF-to-Markdown
Instead of using heavy, unpredictable LLMs for structural recognition, AgenticRAG uses a fast, deterministic, layout-aware Python parser based on `pdfplumber` (`pdf_to_markdown.py`). 
* It extracts visual elements, preserving paragraph separations.
* It parses tabular data into clean, structured **Markdown Tables**, retaining cell alignments and vertical relationships.
* It outputs a cohesive Markdown representation where structural bounds (headings `#`, `##`, `###`) are clearly defined.

### B. Tree Index Construction (`tree_builder.py`)
Instead of splitting the Markdown file into static 500-token chunks, AgenticRAG converts the Markdown file into a **Hierarchical Node Tree** (the `PageIndex`):
* **Node Generation**: Every heading becomes a parent node. Paragraphs and tables immediately following a heading are mapped as child content under that specific heading.
* **Hierarchical Metadata**: Every node is assigned:
  * `doc_id`: A unique hash.
  * `title`: The header text.
  * `parent_id`: Pointer to its structural ancestor (e.g., *Section 1.1* points to *Chapter 1*).
  * `child_ids`: Pointers to its subsections.
  * `sibling_ids`: Horizontal connections to surrounding headers.
* **Semantic Context**: A node retains its parent's headers, giving it structural awareness. For example, a table nested deep in a document knows its full lineage: `Annual Report > Financial Statements > Table 4`.

### C. Graph Storage (`sqlite_graph.py`)
Once the tree is built, it is saved in a local SQLite database (`graph.db` under the default `./pageindex_data` directory).
* **Document Table**: Stores document-level summaries, extracted high-level topics, page counts, and metadata.
* **Nodes Table**: Stores every structural node from the page index (the title, content, level, parent, and siblings).
* **Edges Table**: Stores relational links representing parent-child and sibling hierarchies.
* **SQLite FTS5**: Full-text search indices are built for all node titles, content, summaries, and extracted entities to enable instant structural lookups.

---

## 🔍 2. The Agentic Query Pipeline (Query Stage)

The query pipeline executes a coordinated multi-agent workflow. Rather than returning a flat list of matching passages, the engine uses structural context to hunt down the answer.

### The Problem of Scale
If a document has 800 hierarchical nodes, passing the entire tree structural skeleton to an LLM to select nodes to inspect causes **extreme token bloat** (25,000+ tokens per call) and runs into the **"needle in a haystack"** performance limit. 

### The Solution: Hybrid Sub-Tree Pre-Filtering
Before the expensive LLM-based agents run, AgenticRAG optimizes the tree using local, token-free Python regex:
1. **Keyword Extraction**: The `KeywordAgent` translates the user query into a targeted set of keyphrases and synonyms.
2. **Local Python Regex Scorer**: Every node in the tree is scored locally using Python:
   * **Stem Title Bonus (+10 pts)**: A de-pluralization regex heuristic matches singular/plural differences (e.g. "executives" matching a node titled "Executive Officer").
   * **Deep Text Horizon**: It scans the first **2,000 characters** of each node to detect terms buried inside large tables.
3. **Compact Candidate Sub-Tree**: The top-scoring nodes are retrieved, and their hierarchical ancestors are added back in to preserve structure. This shrinks an 800-node tree into a compact, highly targeted **15-node tree**, saving ~98% of LLM input tokens.

---

## 🤖 3. Deep Dive: The 6 AI Agents

AgenticRAG divides retrieval and synthesis among six highly specialized, task-focused agents. 

---

### 1. The Planner Agent (`planner.py`)
* **Role**: The high-level router and structural query planner.
* **Inputs**: 
  * The user's query (`question`).
  * The history of the chat (`history`).
  * The list of all available document summaries and topics in the `Forest`.
* **Mechanism**:
  * Extracts search terms, key concepts, and document-matching criteria.
  * Formulates a structured JSON plan detailing which documents in the database are likely to contain the answers.
* **Outputs**: A prioritized list of `document_ids` to search, alongside rewritten search-focused queries.

---

### 2. The Keyword Agent (`keyword_agent.py`)
* **Role**: The lexical expander for pre-filtering.
* **Inputs**:
  * The user's query (`question`).
* **Mechanism**:
  * Performs lexical expansion to predict the exact terminology that would appear inside the document's tables, indices, and sections.
  * Generates a JSON array of primary keywords, synonyms, and depluralized stems.
* **Outputs**: A clean JSON list of keywords used to score nodes locally in Python (triggering the **Hybrid Sub-Tree Pre-Filtering**).

---

### 3. The Hunter Agent (`hunter.py`)
* **Role**: The tree explorer.
* **Inputs**:
  * The user's query (`question`).
  * The *Compact Sub-Tree* (hierarchical outline showing only matched candidate nodes, their titles, and parent linkages).
* **Mechanism**:
  * Analyzes the structural skeleton of the compact tree.
  * Acts like a human reader scanning a Table of Contents: it decides which exact nodes are highly likely to contain the specific answer.
  * Employs **reasoning-first execution** to decide whether to fetch a node, its children, or its siblings to gain wider context.
* **Outputs**: A list of specific `node_ids` to retrieve and read in full.

---

### 4. The Evaluator Agent (`evaluator.py`)
* **Role**: The content reader and facts extractor.
* **Inputs**:
  * The user's query (`question`).
  * The full text content of all nodes retrieved by the **Hunter Agent** (including parent context).
* **Mechanism**:
  * Scans the full content of the selected nodes.
  * Extracts raw factual evidence directly related to the user's query.
  * Strips away irrelevant filler text, focusing heavily on tabular data, numbers, and core facts.
* **Outputs**: A structural JSON containing lists of `evidence` statements, each linked to the exact `node_id` and `title` it came from.

---

### 5. The Synthesizer Agent (`synthesizer.py`)
* **Role**: The answer architect.
* **Inputs**:
  * The user's query (`question`).
  * All extracted factual `evidence` blocks compiled by the **Evaluator Agent**.
* **Mechanism**:
  * Consolidates and synthesizes the factual evidence blocks into a clear, cohesive final response.
  * Ensures the answer is perfectly structured, utilizing lists and Markdown tables where appropriate.
* **Outputs**: A draft answer containing explicit inline citations pointing back to the source node headers.

---

### 6. The Critic Agent (`critic.py`)
* **Role**: The zero-hallucination gatekeeper.
* **Inputs**:
  * The user's query (`question`).
  * The draft response from the **Synthesizer Agent**.
  * The raw `evidence` blocks extracted by the **Evaluator Agent**.
* **Mechanism**:
  * Compares every single claim, number, and statement made in the draft answer against the raw source evidence.
  * Runs a strict cross-examination: *Is there any number, date, or assertion in the answer that is not 100% supported by the source text?*
  * If it detects a hallucination, it flags it as unsafe, provides a correction plan, and sends it back to the pipeline to rewrite.
* **Outputs**:
  * `verified`: `True` or `False`.
  * `hallucinations`: A list of any unsupported claims found.
  * `corrected_text`: The finalized, safe answer with all hallucinated claims removed.

---

## 🛡️ 4. How Citations & Zero-Hallucination are Guaranteed

AgenticRAG maintains an unbroken chain of custody for every piece of information:

```
[Raw PDF Layout] ──(Structured Heading)──> [Node ID] ──(Evaluator)──> [Evidence Block] ──(Synthesizer)──> [Final Answer with Citations] ──(Critic Check)──> [User]
```

1. **Explicit Node Paths**: Every segment of text exists within a node path (e.g. `[Section 4.2 > Sub-Table B]`).
2. **Fact Isolation**: The **Evaluator Agent** extracts raw facts *only* and attaches the exact source Node IDs to them.
3. **Rigorous Cross-Checking**: The **Critic Agent** performs mathematical and lexical matching. If the Synthesizer writes *"The company made $12M in Q3"* but the Evaluator's raw evidence reads *"The company made $1.2M in Q3"*, the Critic instantly flags the discrepancy and corrects the output.
4. **Interactive Lineage**: The final answer output includes a `sources` array with exact references, allowing frontend UIs to render clickable citations that jump directly to the correct PDF section.

---

## ⚡ 5. Performance Advantages of the Architecture

| Metric | Traditional Vector RAG | AgenticRAG |
| :--- | :--- | :--- |
| **Indexing Approach** | Arbitrary token splitting (e.g. 500-token chunks with 50-token overlap) | Layout-aware Markdown Tree based on actual heading structures |
| **Search Mechanism** | Vector similarity (Cosine distance over float arrays) | Hierarchical tree navigation & localized regex pre-filtering |
| **Hierarchical Context** | Lost (Deep tables are cut in half, floating in space) | Preserved (deep sub-tables retain their parent heading context) |
| **Token Efficiency** | Poor (Often returns irrelevant chunks that look chemically similar) | **Ultra-efficient** (Prunes ~98% of tree nodes using local regex) |
| **Hallucination Prevention** | None (Relies entirely on LLM generation constraints) | **Guaranteed** (Coordinated Synthesizer-Critic verify-loop) |
| **Accuracy on Numbers/SEC** | High failure rate on detailed data extraction | **Near-perfect** due to tree-preservation of table layouts |
