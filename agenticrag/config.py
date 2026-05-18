"""
config.py — Configuration for PageIndex.

Two config classes:
  - PageIndexConfig : Single-document indexing and search.
  - ForestConfig    : Multi-document Forest (extends PageIndexConfig).

All Groq model IDs are listed as constants in GroqModel so users
get autocomplete and never have to guess a model string.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


class GroqModel:
    """
    Groq-hosted model IDs (as of 2026).
    Pass any of these strings as `model` in PageIndexConfig or ForestConfig.

    Usage:
        config = PageIndexConfig(model=GroqModel.GPT_OSS_20B)
    """

    # ── OpenAI GPT-OSS (via Groq) ─────────────────────────────────────────
    GPT_OSS_20B  = "openai/gpt-oss-20b"   # fast, high quality — recommended default
    GPT_OSS_120B = "openai/gpt-oss-120b"  # largest, best reasoning

    # ── Meta Llama 4 ──────────────────────────────────────────────────────
    LLAMA4_SCOUT = "meta-llama/llama-4-scout-17b-16e-instruct"  # fast multimodal
    LLAMA4_MAVERICK = "meta-llama/llama-4-maverick-17b-128e-instruct"

    # ── Meta Llama 3.3 ────────────────────────────────────────────────────
    LLAMA3_3_70B = "llama-3.3-70b-versatile"  # great general purpose
    LLAMA3_3_8B  = "llama3-8b-8192"           # lightweight / fast

    # ── Qwen 3 ────────────────────────────────────────────────────────────
    QWEN3_32B    = "qwen/qwen3-32b"

    # ── DeepSeek R1 ───────────────────────────────────────────────────────
    DEEPSEEK_R1_DISTILL_LLAMA_70B = "deepseek-r1-distill-llama-70b"

    # ── Kimi ──────────────────────────────────────────────────────────────
    KIMI_K2      = "moonshotai/kimi-k2-instruct"

    # ── Compound (built-in web search) ────────────────────────────────────
    COMPOUND_BETA = "compound-beta"


class LocalModel:
    """
    Model IDs for locally-hosted LLMs via Ollama.
    Set `base_url="http://localhost:11434/v1"` in your config.

    Usage:
        config = ForestConfig(
            model=LocalModel.QWEN3_4B,
            base_url="http://localhost:11434/v1",
        )
    """

    # ── Qwen 3 (recommended for PageIndex) ─────────────────────────
    QWEN3_4B  = "qwen3:4b"     # 2.5GB, 256K ctx — best for ≤5GB VRAM
    QWEN3_8B  = "qwen3:8b"     # 5.2GB, 40K ctx  — best quality/size ratio
    QWEN3_14B = "qwen3:14b"    # 9.3GB, 40K ctx
    QWEN3_30B = "qwen3:30b"    # 19GB, 256K ctx
    QWEN3_32B = "qwen3:32b"    # 20GB, 40K ctx

    # ── Meta Llama ────────────────────────────────────────────
    LLAMA3_2_3B = "llama3.2:3b"  # 2.0GB
    LLAMA3_2_1B = "llama3.2:1b"  # 1.3GB

    # ── Others ────────────────────────────────────────────────
    MISTRAL    = "mistral"       # 4.1GB, 32K ctx
    PHI4       = "phi4"          # 9.1GB, 16K ctx
    GEMMA3_12B = "gemma3:12b"    # 8.1GB, 128K ctx


@dataclass
class PageIndexConfig:
    """
    Configuration for single-document PageIndex operations.

    Parameters
    ----------
    model : str
        Groq model ID.  Use GroqModel constants for convenience.

    api_key : str or None
        Groq API key.  If None, reads from GROQ_API_KEY env var.

    toc_check_pages : int
        Pages to scan for a Table of Contents.

    max_pages_per_node : int
        Maximum page span for a single tree node.

    max_tokens_per_node : int
        Token budget per node window.

    add_node_id : bool
        Assign unique node_id to every node.

    add_node_summary : bool
        Generate a one-sentence summary for each node.

    add_doc_description : bool
        Generate a document-level description.

    add_node_text : bool
        Embed raw text inside each node in the JSON.

    max_retrieval_iterations : int
        How many node-fetch loops the retriever can run.

    temperature : float
        LLM temperature (0.0 = deterministic).

    verbose : bool
        Print progress to stdout.
    """

    model:                   str  = GroqModel.GPT_OSS_20B
    api_key:         Optional[str] = None
    base_url:        Optional[str] = None   # For local LLMs (Ollama, LM Studio, vLLM)

    toc_check_pages:         int  = 20
    max_pages_per_node:      int  = 10
    max_tokens_per_node:     int  = 20_000

    add_node_id:             bool = True
    add_node_summary:        bool = True
    add_doc_description:     bool = True
    add_node_text:           bool = False

    max_retrieval_iterations: int = 5

    enable_pre_filtering:    bool  = True   # Filter large trees via keyword search before LLM loop
    pre_filter_threshold:    int   = 10     # Min node count to activate pre-filtering
    max_filter_candidates:   int   = 3    # Top-N matched nodes kept in candidate sub-tree

    temperature:             float = 0.0
    max_output_tokens:       int   = 4096
    verbose:                 bool  = False
    quiet:                   bool  = False   # Suppress all console output (for web server)
    enable_thinking:         bool  = False   # Enable deep thinking mode (Qwen3 /think)
                                             # True  = slower but higher quality reasoning
                                             # False = faster, skips internal reasoning chain
    num_ctx:                 int   = 32768   # Context window for local LLMs (Ollama)
                                             # KV cache spills from VRAM → RAM automatically
                                             # 32768 = ~2-4GB RAM for 4B models

    # ── Dynamic context-window controls ────────────────────────────────
    # All four limits below were previously hardcoded deep inside agents.
    # Exposing them here lets users with large-context models (e.g.
    # Gemini 2.5 Pro, GPT-OSS 120B) raise them without touching agent code.
    max_chunk_size:          int   = 8000    # Max chars per evidence chunk in the
                                             # Synthesizer.  Raise for wide-context models.
    max_evidence_size:       int   = 64000   # Max total evidence chars sent to Synthesizer.
    max_context_size:        int   = 64000   # Max chars passed to the final answer LLM call
                                             # inside TreeSearcher._answer().
    max_check_size:          int   = 32000   # Max chars used in the sufficiency-check prompt
                                             # inside TreeSearcher._sufficient().
    table_parsing_mode:      bool  = True    # When True, Markdown tables bypass the per-chunk
                                             # truncation cap so no rows are silently dropped
                                             # (e.g. executive officer lists, financial tables).


@dataclass
class ForestConfig(PageIndexConfig):
    """
    Configuration for multi-document Forest operations.

    Extends PageIndexConfig with additional parameters for the
    multi-agent pipeline, storage, and graph backends.

    Parameters
    ----------
    data_dir : str
        Root directory for persisted data (trees, graph DB).

    max_docs_per_query : int
        Maximum documents the Planner can select per query.

    max_hunt_workers : int
        Maximum parallel threads for Hunter agents.

    enable_critic : bool
        Whether to run the Critic (hallucination checker).
        Disable for faster but less safe responses.
    """

    data_dir:           str  = "./pageindex_data"
    max_docs_per_query: int  = 5
    max_hunt_workers:   int  = 5
    parallel_hunting:   bool = True   # If False, hunts documents sequentially (saves TPM)
    enable_critic:      bool = True
    max_retrieval_rounds: int = 3   # Max hunt→synthesize→evaluate cycles
    add_node_text:      bool = True  # Required for Forest to access raw text

    # ── Batch ingestion settings ────────────────────────────────────
    max_batch_workers:    int  = 4    # Parallel PDF→Markdown workers
    max_llm_concurrent:   int  = 2    # Concurrent LLM requests during ingestion