# AgenticRAG System Evaluation Report

This report provides a comprehensive architectural evaluation of **AgenticRAG**, a vectorless, reasoning-based Retrieval-Augmented Generation system. The evaluation is based on a deep read of the system's codebase, including all agents, the tree search algorithms, ingestion pipelines, and storage mechanisms.

## 1. Architectural Overview

AgenticRAG breaks away from the traditional chunk-and-embed paradigm. Instead, it relies on explicit, LLM-driven reasoning over structurally aware document representations. The system uses a multi-agent orchestration pipeline to navigate a document forest, synthesize findings, and rigorously enforce zero-hallucination policies.

### Core Data Structures
*   **PageIndex (The Tree)**: Documents are not chunked arbitrarily. They are parsed into structured Markdown (via `pdf_to_markdown` without LLMs), and then built into a hierarchical JSON tree based on heading levels. This preserves the semantic structure of the document (e.g., Chapter > Section > Subsection), preventing context loss.
*   **DocumentGraph (The Forest)**: Documents are organized as nodes in a graph (defaulting to SQLite + FTS5). This allows for efficient top-level routing based on topics, summaries, and full-text search before the expensive LLM-based tree search begins.

### The Agentic Pipeline (`orchestrator.py`)
The system employs a sophisticated state machine driven by five specialised agents:

1.  **PlannerAgent**: Acts as the router. It extracts search terms and topics from the user's question, queries the DocumentGraph to find candidate documents, and uses an LLM to rank and select the top-N most relevant documents.
2.  **KeywordAgent**: Performs keyword expansion (keyphrases, keywords, synonyms) based on the user's query and the specific context of the document. This is a crucial optimization step for the Hunters.
3.  **HunterAgent & TreeSearcher**: Runs in parallel across the selected documents. 
    *   *Hybrid Pre-filtering*: For large documents, it uses the KeywordAgent's output to perform a fast, pure-Python regex scoring of the tree nodes (scoring titles, summaries, and text). This prunes irrelevant branches and builds a "compact candidate subtree", reducing the LLM's token load by ~95%.
    *   *LLM Selection*: It then prompts the LLM (`SELECT_NODES`) with the compact tree to reason about which specific nodes to read.
    *   It fetches the raw text for the selected nodes and ensures they are relevant before returning them as "chunks".
4.  **SynthesizerAgent**: Takes the retrieved chunks across all parallel Hunters and drafts a comprehensive answer. It enforces strict citation rules (e.g., `[Source: Doc Title, Pages X-Y]`).
5.  **EvaluatorAgent**: The gatekeeper. It evaluates the Synthesizer's draft against the user's question. If the evidence is insufficient, it identifies gaps, refines the query, and triggers another iteration of the Hunter loop (up to a configurable maximum rounds).
6.  **CriticAgent**: The final quality check. It compares the Synthesizer's draft against the *raw source chunks*. If it detects claims that aren't backed by the evidence, it flags them as hallucinations and rewrites the answer to remove them, guaranteeing grounded output.

## 2. Evaluation of System Design

### Strengths
*   **Context Preservation**: By using document structure (headings/trees) instead of arbitrary token-length chunks, the system avoids the "lost in the middle" or "fragmented context" problems common in standard RAG.
*   **Token Efficiency Optimization**: The *Hybrid Sub-Tree Pre-Filtering* is a brilliant engineering choice. Sending a 1,000-node JSON tree to an LLM for every query is prohibitively expensive and slow. By using a lightweight local regex heuristic to prune the tree before the LLM sees it, AgenticRAG achieves the benefits of LLM reasoning without the massive token burn.
*   **Zero-Hallucination Guarantee**: The CriticAgent pattern is robust. By forcing an LLM to verify the draft *strictly* against the retrieved text, the system heavily biases towards accuracy over fluency.
*   **Self-Correction**: The EvaluatorAgent allows the system to realize it missed information and autonomously refine its search query for a second pass. This mirrors human research behavior.
*   **No Vector Database Overhead**: Operating entirely without embeddings or vector databases simplifies deployment significantly. It runs entirely on local file storage and SQLite.

### Trade-offs
*   **Latency**: Even with pre-filtering, the pipeline requires multiple sequential LLM calls (Planner -> Keyword -> Hunter Selection -> Synthesizer -> (Iterate?) -> Critic). This will inherently be slower than a single vector similarity search followed by a single generation call.
*   **Reliance on Good Document Structure**: The `tree_builder.py` relies heavily on the document having a coherent heading structure to build a meaningful tree. If a document is an unstructured wall of text, the tree devolves into a flat list, negating some navigational advantages.

## 3. Comparison with Famous RAG Paradigms

### AgenticRAG vs. Standard Vector RAG (e.g., Basic LangChain/LlamaIndex)
*   **Retrieval Mechanism**: Vector RAG uses Cosine Similarity on dense embeddings. AgenticRAG uses LLM reasoning over a structural Table of Contents.
*   **Context Quality**: Vector RAG often retrieves disjointed paragraphs from different pages. AgenticRAG retrieves logically complete sections (nodes), preserving the narrative and hierarchical context.
*   **Infrastructure**: Vector RAG requires an embedding model and a Vector DB (Chroma, Pinecone, etc.). AgenticRAG requires neither.
*   **Best For**: Vector RAG is better for massive datasets (millions of docs) where you need sub-second retrieval of specific facts. AgenticRAG is vastly superior for complex reasoning tasks across a smaller, curated set of highly structured documents (like financial reports, legal contracts, or manuals).

### AgenticRAG vs. GraphRAG (e.g., Microsoft GraphRAG)
*   **Graph Purpose**: Microsoft GraphRAG extracts granular entities and their relationships (Nodes: "Company A", "Person B"; Edges: "acquired") from the text itself to build a massive knowledge graph, then clusters and summarizes them. It is highly compute-intensive during ingestion.
*   **AgenticRAG Approach**: AgenticRAG only uses a graph at the *document metadata* level (DocumentGraph) for routing. The actual text retrieval is done via Tree Search. 
*   **Best For**: Microsoft GraphRAG excels at global questions ("What are the main themes across all documents?"). AgenticRAG excels at targeted, verifiable reasoning ("Based on the 2018 10-K, what were the risk factors associated with X, and how do they compare to Y?"). AgenticRAG is also vastly cheaper to ingest.

### AgenticRAG vs. Advanced Multi-Step RAG (e.g., ReAct Agents, FLARE)
*   Many advanced RAG frameworks use ReAct loops where an agent decides to use a `Search` tool, reads the output, and searches again.
*   AgenticRAG implements a highly opinionated, domain-specific version of this. Instead of a generic tool-calling loop, it hardcodes the research workflow: Plan -> Hunt (Parallel) -> Synthesize -> Evaluate -> Critic. This constrained pipeline makes it more predictable and easier to tune than a free-form ReAct agent, which can easily spiral into infinite loops or lose track of the original question.

## 4. Conclusion

AgenticRAG represents a highly sophisticated approach to document Q&A. By abandoning vectors in favor of structural trees and explicit LLM reasoning, it solves the context-fragmentation problem that plagues standard RAG. Its multi-agent pipeline is carefully engineered to balance accuracy (via the Critic and Evaluator) with token efficiency (via Hybrid Pre-Filtering). It is exceptionally well-suited for high-stakes domains (legal, financial, technical) where structural context is critical and hallucinations are unacceptable.
