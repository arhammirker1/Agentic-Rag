<p align="center">
  <h1 align="center">AgenticRAG</h1>
  <p align="center"><strong>Vectorless, Reasoning-based RAG for Python</strong></p>
  <p align="center">
    No Vector DB &nbsp;&middot;&nbsp; No Chunking &nbsp;&middot;&nbsp; No Embeddings<br>
    Just pure LLM reasoning over your documents.
  </p>
</p>

<p align="center">
  <a href="#installation">Installation</a> &nbsp;&middot;&nbsp;
  <a href="#quick-start-30-seconds">Quick Start</a> &nbsp;&middot;&nbsp;
  <a href="#web-ui">Web UI</a> &nbsp;&middot;&nbsp;
  <a href="#api-reference">API Reference</a> &nbsp;&middot;&nbsp;
  <a href="#supported-models">Models</a>
</p>

---

**AgenticRAG** is a Python library that lets you ask questions about your PDF documents using AI — without any vector databases, embeddings, or chunking.

Instead of the traditional RAG approach, AgenticRAG:

1. **Builds a smart tree index** from your document (like a Table of Contents, but smarter)
2. **Uses AI agents to reason** over the tree and find the right sections
3. **Verifies every answer** against the source text — zero hallucinations

It works with **Google Gemini**, **Groq Cloud** (free API key) or **local LLMs** via Ollama (100% free, runs on your machine).

---

## Installation

```bash
pip install agentic-rag-core
```

That's it. You're ready to go.

**Optional extras:**

```bash
pip install agentic-rag-core[web]    # Web UI (includes FastAPI server)
pip install agentic-rag-core[gcs]    # Google Cloud Storage backend
pip install agentic-rag-core[neo4j]  # Neo4j graph backend
pip install agentic-rag-core[all]    # Everything
```

> **Note:** If you clone the repo and want to run `server.py`, install with: `pip install -e ".[web]"`

---

## Quick Start (30 seconds)

### Step 1: Get a free API key

Go to [console.groq.com](https://console.groq.com) -- Create an API Key -- Copy it.

### Step 2: Set your API key

Create a `.env` file in your project folder:

```env
GROQ_API_KEY=gsk_your_key_here
```

### Step 3: Ask questions about any PDF

```python
from agenticrag import Forest

# Create a knowledge base and add your PDF
forest = Forest(verbose=True)
forest.add("report.pdf")

# Ask a question
result = forest.ask("What was the net income?")
print(result.text)
```

**That's the entire setup.** Three lines of real code. No vector DB to configure, no embeddings to generate, no chunks to tune.

---

## Use AgenticRAG in Your Own Project

AgenticRAG is designed to be a **drop-in RAG engine** for any Python project. You don't need to understand how RAG works, build any retrieval pipelines, or set up any infrastructure. Just install, import, and ask questions.

### How It Works (The Simple Version)

```
Your app  -->  agenticrag  -->  Answer with sources
               (handles everything:
                PDF parsing, indexing, multi-agent search,
                hallucination checking, citations)
```

You write **zero** retrieval code. AgenticRAG handles all of it internally.

### Example 1: Add Document Q&A to a Flask App

```python
# app.py
from flask import Flask, request, jsonify
from agenticrag import Forest

app = Flask(__name__)

# Create the knowledge base ONCE when the app starts
forest = Forest(verbose=True)
forest.add("company_handbook.pdf")
forest.add("product_docs.pdf")

@app.route("/ask", methods=["POST"])
def ask():
    question = request.json["question"]
    result = forest.ask(question)
    return jsonify({
        "answer": result.text,
        "confidence": result.confidence,
        "sources": result.sources,
    })

if __name__ == "__main__":
    app.run(port=5000)
```

```bash
# Install
pip install agentic-rag-core flask

# Run
python app.py

# Test
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is our refund policy?"}'
```

### Example 2: Build a FastAPI Document API

```python
# api.py
from fastapi import FastAPI
from pydantic import BaseModel
from agenticrag import Forest

app = FastAPI(title="My Document API")
forest = Forest(data_dir="./knowledge_base", verbose=True)

# Add all your documents at startup
forest.add_directory("./docs/", pattern="*.pdf")

class Question(BaseModel):
    text: str

@app.post("/query")
async def query(q: Question):
    result = forest.ask(q.text)
    return {
        "answer": result.text,
        "confidence": result.confidence,
        "pages_used": result.sources,
        "time_seconds": result.elapsed_seconds,
    }
```

```bash
pip install agentic-rag-core fastapi uvicorn
uvicorn api:app --reload
```

### Example 3: Streamlit Chat App (10 lines)

```python
# chat.py
import streamlit as st
from agenticrag import Forest

st.title("Chat with your Documents")

# Initialize once
if "forest" not in st.session_state:
    st.session_state.forest = Forest(verbose=True)
    st.session_state.forest.add("report.pdf")

question = st.text_input("Ask a question:")
if question:
    result = st.session_state.forest.ask(question)
    st.write(result.text)
    st.caption(f"Confidence: {result.confidence:.0%} | {result.elapsed_seconds:.1f}s")
```

```bash
pip install agentic-rag-core streamlit
streamlit run chat.py
```

### Example 4: Simple Python Script

The simplest possible usage — no web framework, no server, just a script:

```python
# ask.py
from agenticrag import Forest

# Point to your documents
forest = Forest()
forest.add("quarterly_report.pdf")
forest.add("annual_report.pdf")

# Ask questions
questions = [
    "What was the total revenue?",
    "What are the main risk factors?",
    "How did expenses change year over year?",
]

for q in questions:
    result = forest.ask(q)
    print(f"\nQ: {q}")
    print(f"A: {result.text}")
    print(f"   Confidence: {result.confidence:.0%}")
    print(f"   Sources: {[s['doc_title'] + ' p.' + s['pages'] for s in result.sources]}")
```

```bash
pip install agentic-rag-core
python ask.py
```

### Example 5: Add to an Existing Django Project

```python
# views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from agenticrag import Forest
import json

# Initialize once — reused across all requests
_forest = Forest(data_dir="./django_knowledge_base")
_forest.add_directory("./media/documents/", pattern="*.pdf")

@csrf_exempt
def document_qa(request):
    if request.method == "POST":
        body = json.loads(request.body)
        result = _forest.ask(body["question"])
        return JsonResponse({
            "answer": result.text,
            "confidence": result.confidence,
        })
```

### Example 6: Use with Local LLMs (100% Free, No API Key)

```python
from agenticrag import Forest, LocalModel

# No API key needed — runs on your machine
forest = Forest(
    model=LocalModel.QWEN3_4B,
    base_url="http://localhost:11434/v1",
)
forest.add("confidential_report.pdf")  # data never leaves your machine
result = forest.ask("What are the projected earnings?")
print(result.text)
```

### Example 7: Build a CLI Tool

```python
# doctool.py
import sys
from agenticrag import Forest

forest = Forest(data_dir="./my_index")

if len(sys.argv) > 2 and sys.argv[1] == "add":
    result = forest.add(sys.argv[2])
    print(f"Indexed: {result.title} ({result.page_count} pages)")

elif len(sys.argv) > 2 and sys.argv[1] == "ask":
    question = " ".join(sys.argv[2:])
    result = forest.ask(question)
    print(f"\n{result.text}\n")
    print(f"Confidence: {result.confidence:.0%} | Sources: {len(result.sources)}")

else:
    print("Usage:")
    print("  python doctool.py add report.pdf")
    print('  python doctool.py ask "What was the revenue?"')
```

```bash
python doctool.py add report.pdf
python doctool.py ask "What was the total revenue in Q4?"
```

### The Key Idea

No matter what you're building — a web app, an API, a chatbot, a CLI tool, or a data pipeline — the pattern is always the same:

```python
from agenticrag import Forest

# 1. Create a Forest (one time)
forest = Forest()

# 2. Add your documents (one time — they're cached)
forest.add("your_file.pdf")

# 3. Ask questions (as many times as you want)
result = forest.ask("your question here")
print(result.text)       # the answer
print(result.confidence) # how sure it is
print(result.sources)    # where it found the info
```

That's it. **Three steps. No RAG architecture knowledge needed.** AgenticRAG handles all the indexing, multi-agent search, synthesis, and hallucination checking internally.

---

## Using AgenticRAG as a Library

### Single Document — `PageIndex`

If you only have one document, use `PageIndex` for the simplest experience:

```python
from agenticrag import PageIndex

# Load and index a PDF (one-time, takes ~30 seconds)
pi = PageIndex()
pi.load("annual_report.pdf")

# Ask questions
result = pi.ask("What was the net income in 2023?")
print(result.text)

# Save the index so you don't have to rebuild it next time
pi.save("report_index.json")

# Later, load it instantly
pi = PageIndex()
pi.load_json("report_index.json")
result = pi.ask("What about revenue?")
```

### Multiple Documents — `Forest`

For multiple documents, use `Forest`. It searches across ALL your documents at once:

```python
from agenticrag import Forest

# Create a knowledge base
forest = Forest(verbose=True)

# Add documents one by one
forest.add("report_2023.pdf")
forest.add("report_2024.pdf")

# Or add an entire folder of PDFs
forest.add_directory("./contracts/", pattern="*.pdf")

# Ask questions across ALL documents
result = forest.ask("Compare revenue growth between 2023 and 2024")

# Rich result object
print(result.text)               # The answer
print(result.confidence)         # 0.0-1.0 confidence score
print(result.sources)            # Which documents & pages were used
print(result.documents_searched) # Which doc IDs were searched
print(result.elapsed_seconds)    # How long it took
```

### Using Local LLMs (Free, No API Key Needed)

You can run AgenticRAG 100% locally with [Ollama](https://ollama.com/download):

```bash
# 1. Install Ollama from https://ollama.com/download
# 2. Pull a model (Qwen3 4B recommended — only 2.5 GB download)
ollama pull qwen3:4b
```

```python
from agenticrag import Forest, LocalModel

forest = Forest(
    model=LocalModel.QWEN3_4B,
    base_url="http://localhost:11434/v1",
    verbose=True,
)
forest.add("report.pdf")
result = forest.ask("What are the key risks?")
print(result.text)
```

### Batch Ingestion (100+ Documents)

For large collections, use the batch pipeline — it's 2-4x faster:

```python
from agenticrag import Forest, LocalModel

forest = Forest(
    model=LocalModel.QWEN3_4B,
    base_url="http://localhost:11434/v1",
    data_dir="./my_knowledge_base",
    verbose=True,
)

# Ingest all PDFs with progress logging
result = forest.add_directory_batch(
    "./papers/",
    pattern="*.pdf",
    resume=True,              # skip already-indexed docs
    skip_description=True,    # halves LLM calls (faster)
    max_llm_concurrent=2,     # concurrent LLM requests
)

print(result)
# BatchResult(total=14000, succeeded=13950, failed=50, skipped=0)

# Now query across all documents
answer = forest.ask("What are the latest findings on X?")
```

### Multi-Turn Conversations

AgenticRAG remembers your conversation automatically:

```python
forest = Forest(verbose=True)
forest.add("report.pdf")

# First question
result = forest.ask("What was the revenue?")
print(result.text)  # "Revenue was $5.2 billion..."

# Follow-up — it remembers the context
result = forest.ask("How does that compare to last year?")
print(result.text)  # "Compared to last year's $4.8 billion..."

# Reset when you want to start fresh
forest.clear_history()
```

---

## Web UI

AgenticRAG includes a beautiful web interface for chatting with your documents:

```bash
# 1. Install the library with web UI dependencies
pip install agentic-rag-core[web]

# 2. Start the web UI server
python -m agenticrag serve

# 3. Start on a custom port
python -m agenticrag serve --port 9000
```

This opens a web app at **http://localhost:8000** where you can:

- Create **notebooks** to organize your documents
- Upload **PDFs** via drag-and-drop
- **Chat** with your documents (with source citations)
- Switch between **Groq Cloud**, **Gemini**, and **local LLM** providers
- Share over **LAN** — anyone on your network can access it

### Split Architecture (GPU on one machine, UI on another)

```bash
# Machine A (GPU): Run Ollama
set OLLAMA_HOST=0.0.0.0
ollama serve

# Machine B (laptop): Start UI and point AgenticRAG to Machine A
python -m agenticrag serve
# Then in Settings > Local LLM > Base URL: http://MACHINE_A_IP:11434/v1
```

---

## API Reference

### High-Level Classes

| Class | What it does | When to use |
|-------|-------------|-------------|
| `Forest` | Multi-document knowledge base | **Most common** — use this for everything |
| `PageIndex` | Single-document index | When you only have one document |
| `ForestResult` | Result from `Forest.ask()` | Access `.text`, `.sources`, `.confidence` |

### `Forest` Methods

| Method | Description |
|--------|-------------|
| `Forest(model=..., verbose=True)` | Create a new knowledge base |
| `.add("file.pdf")` | Add a single document |
| `.add_directory("./docs/")` | Add all PDFs from a folder |
| `.add_directory_batch("./docs/")` | Fast batch add (100+ docs) |
| `.ask("question")` | Ask a question across all docs |
| `.documents()` | List all indexed documents |
| `.remove(doc_id)` | Remove a document |
| `.size` | Number of documents |
| `.clear_history()` | Reset conversation memory |
| `.info()` | Forest status summary |

### `PageIndex` Methods

| Method | Description |
|--------|-------------|
| `PageIndex(model=..., verbose=True)` | Create a new single-doc index |
| `.load("file.pdf")` | Index a document |
| `.save("index.json")` | Save the index to disk |
| `.load_json("index.json")` | Load a saved index |
| `.ask("question")` | Ask a question |

### `ForestResult` Fields

| Field | Type | Description |
|-------|------|-------------|
| `.text` | `str` | The final verified answer |
| `.confidence` | `float` | 0.0 to 1.0 confidence score |
| `.sources` | `list` | Which documents/pages were used |
| `.documents_searched` | `list` | Which doc IDs were searched |
| `.reasoning_trace` | `list` | Step-by-step agent pipeline trace |
| `.was_rewritten` | `bool` | Whether the Critic modified the answer |
| `.hallucinations` | `list` | Any hallucinations that were caught |
| `.elapsed_seconds` | `float` | Total time taken |

### Configuration

```python
from agenticrag import ForestConfig, GroqModel

config = ForestConfig(
    model              = GroqModel.GPT_OSS_20B,  # Which AI model to use
    data_dir           = "./my_data",             # Where to store indices
    max_docs_per_query = 5,                       # Max docs to search per question
    max_hunt_workers   = 5,                       # Parallel search threads
    enable_critic      = True,                    # Zero-hallucination checking
    verbose            = True,                    # Print progress
    # ── Hybrid Sub-Tree Pre-Filtering ──────────────────────────────────
    # Activates for trees larger than pre_filter_threshold nodes.
    # One small KeywordAgent LLM call (256 tokens) replaces up to 10
    # large SELECT_NODES calls (25,000 tokens each) on big documents.
    enable_pre_filtering = True,   # Toggle the entire pre-filter pipeline
    pre_filter_threshold = 50,     # Min node count to activate (default: 50)
    max_filter_candidates = 20,    # Top-N matched seed nodes (default: 20)
)
```

Or pass these directly to `Forest()`:

```python
forest = Forest(
    model=GroqModel.GPT_OSS_20B,
    data_dir="./my_data",
    verbose=True,
    enable_pre_filtering=True,   # default — disable if you want full tree always
    pre_filter_threshold=50,     # lower = activates on smaller documents
    max_filter_candidates=20,    # higher = more candidate nodes, larger prompt
)
```

---

## Supported Models

### Cloud Models (Groq — Free API)

```python
from agenticrag import Forest, GroqModel

forest = Forest(model=GroqModel.GPT_OSS_20B)      # Fast, recommended default
forest = Forest(model=GroqModel.GPT_OSS_120B)      # Largest, best reasoning
forest = Forest(model=GroqModel.LLAMA4_SCOUT)       # Llama 4 Scout
forest = Forest(model=GroqModel.LLAMA3_3_70B)       # Llama 3.3 70B
forest = Forest(model=GroqModel.QWEN3_32B)          # Qwen 3 32B
forest = Forest(model=GroqModel.DEEPSEEK_R1_DISTILL_LLAMA_70B)  # DeepSeek R1
```

### Local Models (Ollama — 100% Free)

```bash
# Install from https://ollama.com/download, then:
ollama pull qwen3:4b     # 2.5 GB — recommended
ollama pull qwen3:8b     # 5.2 GB — better quality
ollama pull qwen3:14b    # 9.3 GB — even better
```

```python
from agenticrag import Forest, LocalModel

forest = Forest(
    model=LocalModel.QWEN3_4B,
    base_url="http://localhost:11434/v1",
)
```

| Model | Download Size | VRAM Needed | Best For |
|-------|-------------|-------------|----------|
| `LocalModel.QWEN3_4B` | 2.5 GB | 5 GB or less | Low-VRAM GPUs, fastest |
| `LocalModel.QWEN3_8B` | 5.2 GB | 8 GB or less | Best quality/size ratio |
| `LocalModel.QWEN3_14B` | 9.3 GB | 12 GB or less | Higher quality |
| `LocalModel.QWEN3_30B` | 19 GB | 24 GB or less | Strong reasoning |
| `LocalModel.LLAMA3_2_3B` | 2.0 GB | 4 GB or less | Ultra-lightweight |
| `LocalModel.MISTRAL` | 4.1 GB | 6 GB or less | General purpose |
| `LocalModel.GEMMA3_12B` | 8.1 GB | 12 GB or less | Alternative mid-range |

---

## How It Works

AgenticRAG uses a **multi-agent pipeline** — like a team of AI researchers working together:

```
Your Question
     |
     v
+---------+   Looks at the document graph to find
| Planner |-->  which documents might have the answer
+---------+
     |
     v
+---------+   Searches those documents IN PARALLEL
| Hunters |-->  using tree-based reasoning (not keywords!)
+---------+
     |
     v
+--------------+   Combines evidence from multiple docs
| Synthesizer  |-->  into a single, coherent answer
+--------------+
     |
     v
+--------+   Checks every claim against the source text
| Critic |-->  removes anything not backed by evidence
+--------+
     |
     v
  Verified Answer
```

This is why AgenticRAG can answer complex questions across many documents — it doesn't just find similar text, it actually **reasons** about what's relevant.

---

## Comparison: AgenticRAG vs. Vector RAG

AgenticRAG is not meant to replace Vector RAG for all use cases. It is designed specifically for **deep research on highly structured documents** (like SEC filings, legal contracts, or technical manuals) where exact facts and numbers matter.

### When to use Vector RAG
* **Goal**: Finding general semantic themes across millions of documents (e.g., "What is the company's general tone regarding AI?").
* **Why**: Vector RAG is fast, cheap, and scales to millions of documents instantly. It uses cosine similarity to find semantically related text. 
* **Weakness**: It blindly chunks documents, often splitting tables or separating numbers from their context. It struggles with exact keyword constraints and is prone to LLM hallucinations when synthesizing answers from disjointed chunks.

### When to use AgenticRAG
* **Goal**: Targeted, verifiable reasoning on a curated set of complex documents (e.g., "What was the exact amortisation of intangible assets in 2023?").
* **Why**: 
  1. **Context Preservation**: It preserves the document's structure as a tree. Financial tables remain intact under their logical headings.
  2. **Exact Retrieval**: The `KeywordAgent` combined with regex pre-filtering ensures the system hunts for exact strings (like "Operating Margin") rather than mathematically similar concepts.
  3. **Zero-Hallucination**: The `CriticAgent` strictly cross-references the drafted answer against the raw retrieved text, removing any hallucinated numbers.
* **Weakness**: It is token-heavy and slow (multiple LLM calls per query). It is not built for high-throughput, low-latency web search over millions of documents.

---

## Hybrid Sub-Tree Pre-Filtering

For large documents (SEC 10-K filings, legal contracts, technical manuals), the
tree index can contain hundreds of nodes. Passing the entire tree into every
SELECT_NODES call is slow, expensive, and causes the LLM to miss relevant nodes
(needle-in-a-haystack problem).

AgenticRAG automatically activates **Hybrid Sub-Tree Filtering** when a tree
exceeds `pre_filter_threshold` nodes:
User Question
│
▼
KeywordAgent.expand()          — 1 LLM call, 256 tokens max
• Generates keyphrases, keywords, synonyms
• Receives document_description so it predicts
vocabulary specific to THIS document
│
▼
_local_node_search()           — 0 LLM calls, pure Python regex
• Applies Stem Title Bonus (+10 pts) for depluralised keyword matches
• Scores exact hits across title, summary, and deep text-preview (2000 chars)
• Returns node IDs ranked by hit count + bonuses
│
▼
_build_candidate_subtree()     — 0 LLM calls
• Keeps top-N matched nodes + all their ancestors
• Preserves hierarchical context for the LLM
│
▼
SELECT_NODES loop              — on ~15-node sub-tree, not 1,000-node full tree
• ~500 tokens per call instead of ~25,000 tokens
• 98% token reduction on large documents

### Token savings example (3M 2018 10-K, ~160 pages)

| Metric | Without Pre-Filter | With Pre-Filter |
|--------|-------------------|-----------------|
| Nodes passed to LLM | ~800 | ~15 |
| Tokens per SELECT_NODES call | ~20,000 | ~400 |
| Total tokens (3 iterations) | ~60,000 | ~1,200 |
| Rate-limit risk | High | Negligible |

### Developer logging

Set `verbose=True` on your `Forest` to see the full pre-filter trace in the
console. All steps are also written to `pageindex_data/logs/trail.log`:

==================== PRE-FILTER INITIATED ====================
INFO: Tree size (800 nodes) exceeds threshold (50).
Running keyword pre-filtering to build a compact candidate sub-tree.
==================== KEYWORD AGENT (INPUT) ====================
DATA: { "question": "what distributions do we have?",
"doc_context_preview": "This document is 3M Company's 2018 Annual Report..." }
==================== KEYWORD AGENT (SUCCESS) ====================
DATA: { "expanded_keywords": ["distributions", "dividends paid", "financing activities",
"treasury stock", "stockholder payouts", ...] }
==================== PRE-FILTER COMPLETED ====================
INFO: Filtered 800 nodes → 18 candidate seed nodes + their ancestors.
DATA: { "candidate_seed_ids": ["0042", "0043", "0089", ...],
"matched_keywords": ["dividends paid", "financing activities", ...] }

### Configuration reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_pre_filtering` | `True` | Toggle the entire pipeline on/off |
| `pre_filter_threshold` | `50` | Min node count before activation |
| `max_filter_candidates` | `20` | Max seed nodes passed to sub-tree builder |

**When to tune these:**
- Slow responses on small documents → raise `pre_filter_threshold` to 100+
- Missing relevant nodes on huge documents → raise `max_filter_candidates` to 30-40
- Pre-filter gives zero matches repeatedly → check `trail.log` for the expanded keywords

---

## Project Structure
```
agenticrag/
|-- __init__.py              # Public API (Forest, PageIndex, etc.)
|-- __main__.py              # python -m agenticrag entry point
|-- config.py                # GroqModel, LocalModel, configuration
|-- groq_client.py           # LLM wrapper (Groq + OpenAI-compatible)
|-- pdf_parser.py            # PDF/Markdown text extraction
|-- pdf_to_markdown.py       # PDF to Markdown converter (no LLM)
|-- prompts.py               # All LLM prompts
|-- tree_builder.py          # Build hierarchical tree index
|-- tree_search.py           # Single-document tree search
|-- pageindex.py             # PageIndex (single-doc wrapper)
|-- forest.py                # Forest (multi-doc entry point)
|-- agents/
|   |-- planner.py           # Document selection from graph
|   |-- hunter.py            # Parallel document searching
|   |-- synthesizer.py       # Multi-doc answer synthesis
|   |-- evaluator.py         # Retrieval sufficiency checking
|   |-- critic.py            # Zero-hallucination enforcer
|   +-- orchestrator.py      # Agentic loop state machine
|-- storage/
|   |-- base.py              # Abstract TreeStore interface
|   |-- local.py             # Local filesystem (default)
|   +-- gcs.py               # Google Cloud Storage
|-- graph/
|   |-- base.py              # Abstract DocumentGraph interface
|   |-- sqlite_graph.py      # SQLite + FTS5 (default)
|   +-- neo4j_graph.py       # Neo4j (production scale)
+-- ingestion/
    |-- metadata.py          # LLM metadata extraction
    |-- pipeline.py          # Single-document ingestion
    +-- batch.py             # Batch ingestion (100K+ docs)
```

---

## Storage Backends

### Local Filesystem (Default)

```python
# Automatic — just use Forest() and it stores in ./pageindex_data/
forest = Forest()
```

### Google Cloud Storage

```bash
pip install agentic-rag-core[gcs]
```

```python
from agenticrag import Forest
from agenticrag.storage import GCSStore

store = GCSStore(
    bucket_name="my-bucket",
    prefix="trees/",
    credentials="path/to/service-account.json",
)
forest = Forest(store=store)
```

### Neo4j Graph (Production)

```bash
pip install agentic-rag-core[neo4j]
```

```python
from agenticrag import Forest
from agenticrag.graph import Neo4jGraph

graph = Neo4jGraph(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="your_password",
)
forest = Forest(graph=graph)
```

---

## Tips and Best Practices

| Tip | Why |
|-----|-----|
| Start with `Forest()` | Works out of the box, zero config |
| Use `verbose=True` | See exactly what the AI agents are doing |
| Use batch ingestion for 100+ docs | `forest.add_directory_batch()` is 2-4x faster |
| Set `skip_description=True` in batch | Halves the number of LLM calls |
| Use `resume=True` in batch | Safely restart interrupted runs |
| Use `skip_critic=True` for speed | Faster answers (but less hallucination protection) |
| Try Qwen3 4B for local LLMs | Best quality-per-VRAM model available (2.5 GB) |

---

## Hardware Guide (Local LLMs)

| Setup | GPU | VRAM | Recommended Model | Speed |
|-------|-----|------|--------------------|-------|
| **Minimum** | Any NVIDIA (CUDA >= 5.0) | 4-5 GB | Qwen3 4B | Good |
| **Good** | RTX 3060 / Quadro P2000 | 5-12 GB | Qwen3 4B or 8B | Better |
| **Recommended** | RTX 4070 Ti | 16 GB | Qwen3 8B or 14B | Fast |
| **Ideal** | RTX 4090 | 24 GB | Qwen3 30B | Fastest |

> **Note:** AMD GPUs with Polaris/GCN architecture (like RX 580) are **not supported** by Ollama. Only NVIDIA GPUs with CUDA >= 5.0 and AMD RDNA (RX 5000+) work.

---

## FAQ

<details>
<summary><strong>What file types are supported?</strong></summary>

PDF, Markdown (.md), and plain text (.txt) files.

</details>

<details>
<summary><strong>Do I need a GPU?</strong></summary>

**No.** If you use Groq Cloud (free API), everything runs in the cloud. You only need a GPU if you want to run local LLMs via Ollama.

</details>

<details>
<summary><strong>How is this different from LangChain / LlamaIndex?</strong></summary>

Traditional RAG (LangChain, LlamaIndex) splits documents into chunks and uses vector similarity to find relevant pieces. This breaks down on professional documents because **similarity is not the same as relevance**.

AgenticRAG builds a hierarchical tree index and uses LLM reasoning to navigate it — like a human expert flipping through a report. No vectors, no embeddings, no chunking.

</details>

<details>
<summary><strong>How do I get a Groq API key?</strong></summary>

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up (free)
3. Go to API Keys > Create API Key
4. Copy the key that starts with `gsk_`

</details>

<details>
<summary><strong>Can I use OpenAI / Anthropic / other providers?</strong></summary>

AgenticRAG works with any OpenAI-compatible API. Set the `base_url` parameter to point to your provider's endpoint.

</details>

---

## License

MIT — free to use in personal and commercial projects.

---

## Credits

Architecture inspired by [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex).

Built by [Arham Mirkar](https://github.com/AjayVirkar).
