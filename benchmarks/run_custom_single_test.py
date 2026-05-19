#!/usr/bin/env python3
"""
run_custom_single_test.py -- Head-to-Head Single-Question RAG Benchmark
Compares Vector RAG (sentence-transformers + ChromaDB) vs AgenticRAG (Forest)
on a single PDF and question with LLM-as-a-Judge evaluation.

Usage:
    python run_custom_single_test.py
    # Or with parameters:
    python run_custom_single_test.py --pdf my_doc.pdf --question "..." --gold-answer "..."
"""

import argparse
import json
import os
import sys
import time
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Setup path and environment
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# Enable ANSI colors on Windows terminal
if sys.platform == 'win32':
    os.system('')

# Colors for terminal styling
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Get Groq API Key
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
if not GROQ_API_KEY:
    print(f"{Colors.RED}{Colors.BOLD}ERROR:{Colors.ENDC} Set GROQ_API_KEY in your .env file.")
    sys.exit(1)

def next_key() -> str:
    return GROQ_API_KEY


# ---------------------------------------------------------------------------
# Vector RAG baseline implementation
# ---------------------------------------------------------------------------
def _split_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Simple recursive text splitter (no LangChain dependency)."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            for sep in ["\n\n", "\n", ". ", " "]:
                last_sep = text.rfind(sep, start, end)
                if last_sep > start:
                    end = last_sep + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if end < len(text) else len(text)
    return chunks


def build_vector_baseline(pdf_path: Path) -> Any:
    """Build a ChromaDB vector store from a single PDF."""
    import chromadb
    from sentence_transformers import SentenceTransformer

    print(f"\n{Colors.CYAN}{Colors.BOLD}[Vector RAG] Building Vector Store (ChromaDB + MiniLM)...{Colors.ENDC}")
    print("  Loading embeddings model (sentence-transformers/all-MiniLM-L6-v2)...")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    all_chunks = []
    all_metadatas = []
    all_ids = []

    print(f"  Extracting text and chunking PDF: {pdf_path.name}...")
    try:
        import pdfplumber
        doc_chunk_count = 0
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if len(text.strip()) < 20:
                    continue
                chunks = _split_text(text)
                for ci, chunk in enumerate(chunks):
                    chunk_id = f"p{page_num}_c{ci}"
                    all_chunks.append(chunk)
                    all_metadatas.append({
                        "page": str(page_num),
                        "source": pdf_path.name,
                    })
                    all_ids.append(chunk_id)
                    doc_chunk_count += 1
        print(f"  Extracted {Colors.GREEN}{doc_chunk_count}{Colors.ENDC} chunks.")
    except Exception as e:
        print(f"{Colors.RED}ERROR parsing PDF: {e}{Colors.ENDC}")
        sys.exit(1)

    if not all_chunks:
        print(f"{Colors.RED}ERROR: No text could be extracted from this PDF.{Colors.ENDC}")
        sys.exit(1)

    print(f"  Generating embeddings for {len(all_chunks)} chunks...")
    embeddings = model.encode(all_chunks, show_progress_bar=True, batch_size=64)

    print(f"  Initializing ChromaDB collection...")
    chroma_dir = Path(__file__).parent / "_chroma_db_single"
    if chroma_dir.exists():
        import shutil
        shutil.rmtree(chroma_dir)

    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.create_collection(
        name="single_test_benchmark",
        metadata={"hnsw:space": "cosine"},
    )
    
    collection.add(
        ids=all_ids,
        documents=all_chunks,
        metadatas=all_metadatas,
        embeddings=embeddings.tolist(),
    )

    print(f"  Vector store ready: {Colors.GREEN}{collection.count()}{Colors.ENDC} vectors.")
    return {"collection": collection, "model": model}


def query_vector_rag(
    vectorstore: Dict,
    question: str,
    api_key: str,
    model: str = "llama-3.3-70b-versatile",
) -> str:
    """Query the vector RAG baseline."""
    collection = vectorstore["collection"]
    model_embed = vectorstore["model"]

    # Embed the question and retrieve top-5
    q_embedding = model_embed.encode([question])[0].tolist()
    results = collection.query(query_embeddings=[q_embedding], n_results=5)

    context = "\n\n---\n\n".join(results["documents"][0])

    from groq import Groq
    client = Groq(api_key=api_key)

    prompt = (
        "You are an expert analyst. Answer the following question using ONLY "
        "the provided context. If the context does not contain enough information, "
        "say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1000,
    )

    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Agentic RAG runner implementation
# ---------------------------------------------------------------------------
def build_agenticrag(pdf_path: Path, model: str = "llama-3.3-70b-versatile", enable_critic: bool = True) -> Any:
    """Build an AgenticRAG Forest from the single PDF."""
    from agenticrag import Forest

    print(f"\n{Colors.CYAN}{Colors.BOLD}[AgenticRAG] Building AgenticRAG Forest...{Colors.ENDC}")
    data_dir = Path(__file__).parent / "_agenticrag_data_single"

    forest = Forest(
        model=model,
        api_key=next_key(),
        data_dir=str(data_dir),
        verbose=True,
        parallel_hunting=True,
        enable_critic=enable_critic,
    )

    doc_name = pdf_path.stem
    try:
        existing = forest.documents()
        already = any(doc_name in (d.get("file_name", "") or d.get("title", ""))
                      for d in existing)
        if already:
            print(f"  [cached] Document {Colors.GREEN}{pdf_path.name}{Colors.ENDC} is already indexed in Forest.")
            return forest
    except Exception:
        pass

    print(f"  Indexing document in Forest: {pdf_path.name}...")
    forest.add(str(pdf_path))
    return forest


# ---------------------------------------------------------------------------
# LLM-as-a-Judge Prompt Templates & Handlers
# ---------------------------------------------------------------------------
JUDGE_GOLD_PROMPT = """You are an impartial evaluator for a RAG (Retrieval-Augmented Generation) system benchmark.

You will be given:
- A QUESTION about a document
- The GOLD ANSWER (correct answer verified by human experts)
- The SYSTEM ANSWER (generated by the system being evaluated)

Score the SYSTEM ANSWER on two dimensions:

1. CORRECTNESS (1-5): How accurately does the system answer match the gold answer?
   1 = Completely wrong or irrelevant
   2 = Partially addresses the question but key facts are wrong
   3 = Addresses the question with some correct info but misses key details
   4 = Mostly correct with minor omissions
   5 = Fully correct, matches the gold answer

2. FAITHFULNESS (1-5): Is the system answer free of hallucinated information?
   1 = Mostly hallucinated / fabricated information
   2 = Contains significant unsupported claims
   3 = Some claims may not be grounded in evidence
   4 = Mostly grounded, minor unsupported details
   5 = Fully grounded, no hallucinations detected

Respond in this EXACT JSON format and nothing else:
{{"correctness": <int>, "faithfulness": <int>, "reasoning": "<brief explanation>"}}

---

QUESTION: {question}

GOLD ANSWER: {gold_answer}

SYSTEM ANSWER: {system_answer}"""

JUDGE_COMPARATIVE_PROMPT = """You are an expert impartial evaluator comparing two RAG (Retrieval-Augmented Generation) systems head-to-head on the same question.

You will be given:
- The QUESTION
- The ANSWER from System A (Vector RAG)
- The ANSWER from System B (Agentic RAG)

Compare both answers on the following parameters:
1. COMPLETENESS & DETAIL: Which answer covers the question more fully?
2. ACCURACY & CONCISENESS: Which answer is more precise and directly answers the prompt without fluff?
3. TRUTHFULNESS & GROUNDING: Which answer feels more reliable, factually detailed, and grounded (e.g. referencing specific numbers, pages, or sections if available)?

Score each system 1 to 5:
- Score 1: Completely wrong, irrelevant, or highly hallucinated.
- Score 3: Partially correct, misses key details, or too generic.
- Score 5: Exceptionally precise, comprehensive, fully correct, and perfectly grounded.

Explain your comparative reasoning, detailing why you scored them the way you did, highlighting strengths and weaknesses of each.

Respond in this EXACT JSON format and nothing else:
{{
  "vector_rag": {{
    "correctness": <int>,
    "reasoning": "<evaluation of Vector RAG strengths/weaknesses>"
  }},
  "agentic_rag": {{
    "correctness": <int>,
    "reasoning": "<evaluation of Agentic RAG strengths/weaknesses>"
  }},
  "comparison_summary": "<overall summary of which performed better and why>"
}}

---

QUESTION: {question}

SYSTEM A (Vector RAG) ANSWER:
{vector_answer}

SYSTEM B (Agentic RAG) ANSWER:
{agentic_answer}"""


def judge_answer_with_gold(
    question: str,
    gold_answer: str,
    system_answer: str,
    api_key: str,
    model: str = "llama-3.3-70b-versatile",
) -> Tuple[int, int, str]:
    from groq import Groq

    if not system_answer or len(system_answer.strip()) < 5:
        return 1, 1, "Empty or trivial answer"

    client = Groq(api_key=api_key)
    prompt = JUDGE_GOLD_PROMPT.format(
        question=question,
        gold_answer=gold_answer,
        system_answer=system_answer,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=400,
        )
        text = response.choices[0].message.content.strip()
        json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return (
                min(5, max(1, int(data.get("correctness", 1)))),
                min(5, max(1, int(data.get("faithfulness", 1)))),
                data.get("reasoning", ""),
            )
        return 3, 3, f"Could not parse judge output: {text[:200]}"
    except Exception as e:
        return 3, 3, f"Judge error: {e}"


def judge_comparative(
    question: str,
    vector_answer: str,
    agentic_answer: str,
    api_key: str,
    model: str = "llama-3.3-70b-versatile",
) -> Dict[str, Any]:
    from groq import Groq

    client = Groq(api_key=api_key)
    prompt = JUDGE_COMPARATIVE_PROMPT.format(
        question=question,
        vector_answer=vector_answer,
        agentic_answer=agentic_answer,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=700,
        )
        text = response.choices[0].message.content.strip()
        json_match = re.search(r'\{.+\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {
            "vector_rag": {"correctness": 3, "reasoning": "Could not parse judge output."},
            "agentic_rag": {"correctness": 3, "reasoning": "Could not parse judge output."},
            "comparison_summary": f"Could not parse judge output: {text[:200]}"
        }
    except Exception as e:
        return {
            "vector_rag": {"correctness": 1, "reasoning": f"Judge error: {e}"},
            "agentic_rag": {"correctness": 1, "reasoning": f"Judge error: {e}"},
            "comparison_summary": f"Error running comparative judge: {e}"
        }


# ---------------------------------------------------------------------------
# Cache clearing utility
# ---------------------------------------------------------------------------
def clear_caches():
    import shutil
    chroma_dir = Path(__file__).parent / "_chroma_db_single"
    agentic_dir = Path(__file__).parent / "_agenticrag_data_single"
    print(f"\n{Colors.YELLOW}{Colors.BOLD}Clearing RAG caches for a fresh run...{Colors.ENDC}")
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)
        print("  - Vector RAG ChromaDB cache deleted.")
    if agentic_dir.exists():
        shutil.rmtree(agentic_dir)
        print("  - AgenticRAG Forest database deleted.")


# ---------------------------------------------------------------------------
# CLI & Execution
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Head-to-head comparison of Vector RAG vs AgenticRAG on a single question.",
    )
    parser.add_argument("--pdf", type=str, help="Path to local PDF document")
    parser.add_argument("--question", type=str, help="The query question")
    parser.add_argument("--gold-answer", type=str, help="Optional expected reference answer")
    parser.add_argument("--model", type=str, default="llama-3.3-70b-versatile", help="Model used for RAG generation and judge evaluation")
    parser.add_argument("--skip-vector", action="store_true", help="Skip Vector RAG baseline test")
    parser.add_argument("--clear-cache", action="store_true", help="Clear all stored indexing caches for a completely fresh run")
    parser.add_argument("--disable-critic", action="store_true", help="Disable Critic agent validation in AgenticRAG")
    args = parser.parse_args()

    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 75}")
    print(f"  RAG Head-to-Head Benchmark: Vector RAG vs AgenticRAG")
    print(f"{'=' * 75}{Colors.ENDC}")

    # Clear caches if requested
    if args.clear_cache:
        clear_caches()

    # Interactive flow if parameters are missing
    is_interactive = not (args.pdf and args.question)
    pdf_path_str = args.pdf
    question = args.question
    gold_answer = args.gold_answer

    if not pdf_path_str:
        print(f"\n{Colors.CYAN}{Colors.BOLD}PDF File Setup:{Colors.ENDC}")
        while True:
            inp = input("  Enter the path to your PDF file: ").strip()
            # Clean quotes if user dragged and dropped file
            inp = inp.strip('"').strip("'")
            if not inp:
                continue
            path_test = Path(inp)
            if path_test.is_file() and path_test.suffix.lower() == '.pdf':
                pdf_path_str = inp
                break
            else:
                print(f"  {Colors.RED}Invalid file. Make sure the file exists and is a PDF.{Colors.ENDC}")

    pdf_path = Path(pdf_path_str).resolve()

    if not question:
        print(f"\n{Colors.CYAN}{Colors.BOLD}Question Setup:{Colors.ENDC}")
        while True:
            question = input("  Enter the question to test: ").strip()
            if question:
                break

    if is_interactive and gold_answer is None:
        print(f"\n{Colors.CYAN}{Colors.BOLD}Expected Answer Setup (Optional):{Colors.ENDC}")
        gold_answer = input("  Enter the expected gold answer (or press [Enter] to skip): ").strip()

    # 1. Build and Run Vector RAG
    vector_ans = ""
    vector_time = 0.0
    if not args.skip_vector:
        t0 = time.time()
        try:
            vectorstore = build_vector_baseline(pdf_path)
            print(f"\n{Colors.CYAN}  [Vector RAG] Running question...{Colors.ENDC}")
            t_query0 = time.time()
            vector_ans = query_vector_rag(vectorstore, question, next_key(), model=args.model)
            vector_time = time.time() - t_query0
            print(f"  Generated answer in {Colors.GREEN}{vector_time:.2f}s{Colors.ENDC}")
        except Exception as e:
            vector_ans = f"Error: {e}"
            print(f"  {Colors.RED}Vector RAG failed: {e}{Colors.ENDC}")

    # 2. Build and Run AgenticRAG
    agentic_ans = ""
    agentic_time = 0.0
    t0 = time.time()
    try:
        forest = build_agenticrag(pdf_path, model=args.model, enable_critic=not args.disable_critic)
        print(f"\n{Colors.CYAN}  [AgenticRAG] Running question...{Colors.ENDC}")
        t_query0 = time.time()
        result = forest.ask(question, skip_critic=args.disable_critic)
        agentic_ans = result.text
        agentic_time = time.time() - t_query0
        print(f"  Generated answer in {Colors.GREEN}{agentic_time:.2f}s{Colors.ENDC}")
    except Exception as e:
        agentic_ans = f"Error: {e}"
        print(f"  {Colors.RED}AgenticRAG failed: {e}{Colors.ENDC}")

    # 3. LLM-as-a-Judge Evaluation
    print(f"\n{Colors.HEADER}{Colors.BOLD}[LLM-as-a-Judge] Grading responses...{Colors.ENDC}")
    
    has_gold = bool(gold_answer and gold_answer.strip())
    judge_model = args.model

    vector_score = 0
    vector_faith = 0
    vector_reason = ""
    agentic_score = 0
    agentic_faith = 0
    agentic_reason = ""
    comp_summary = ""

    if has_gold:
        print("  Evaluating answers against the provided Gold Answer...")
        if not args.skip_vector:
            c, f, r = judge_answer_with_gold(question, gold_answer, vector_ans, next_key(), model=judge_model)
            vector_score, vector_faith, vector_reason = c, f, r
        
        c, f, r = judge_answer_with_gold(question, gold_answer, agentic_ans, next_key(), model=judge_model)
        agentic_score, agentic_faith, agentic_reason = c, f, r
    else:
        print("  Evaluating answers using comparative head-to-head grading (no gold answer)...")
        if not args.skip_vector:
            res = judge_comparative(question, vector_ans, agentic_ans, next_key(), model=judge_model)
            vector_score = res.get("vector_rag", {}).get("correctness", 1)
            vector_reason = res.get("vector_rag", {}).get("reasoning", "")
            agentic_score = res.get("agentic_rag", {}).get("correctness", 1)
            agentic_reason = res.get("agentic_rag", {}).get("reasoning", "")
            comp_summary = res.get("comparison_summary", "")

    # 4. Present Beautiful CLI Comparison
    from tabulate import tabulate

    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 75}")
    print(f"  COMPARATIVE BENCHMARK SUMMARY")
    print(f"{'=' * 75}{Colors.ENDC}")

    table_data = []
    
    # Format AgenticRAG row
    agentic_score_str = f"{agentic_score}/5"
    agentic_faith_str = f"{agentic_faith}/5" if has_gold else "N/A"
    table_data.append([
        f"{Colors.GREEN}{Colors.BOLD}Agentic RAG{Colors.ENDC}",
        f"{Colors.GREEN}{agentic_time:.2f}s{Colors.ENDC}",
        f"{Colors.GREEN}{agentic_score_str}{Colors.ENDC}",
        f"{Colors.GREEN}{agentic_faith_str}{Colors.ENDC}"
    ])

    # Format Vector RAG row
    if not args.skip_vector:
        vector_score_str = f"{vector_score}/5"
        vector_faith_str = f"{vector_faith}/5" if has_gold else "N/A"
        table_data.append([
            f"{Colors.BLUE}{Colors.BOLD}Vector RAG{Colors.ENDC}",
            f"{Colors.BLUE}{vector_time:.2f}s{Colors.ENDC}",
            f"{Colors.BLUE}{vector_score_str}{Colors.ENDC}",
            f"{Colors.BLUE}{vector_faith_str}{Colors.ENDC}"
        ])

    headers = ["RAG System", "Latency", "Quality Score", "Faithfulness"]
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

    # Print Answers Side-by-side or stacked
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'-' * 75}")
    print(f"Generated Answers:")
    print(f"{'-' * 75}{Colors.ENDC}")
    
    if not args.skip_vector:
        print(f"\n{Colors.BLUE}{Colors.BOLD}[Vector RAG Answer]:{Colors.ENDC}")
        print(vector_ans)
        
    print(f"\n{Colors.GREEN}{Colors.BOLD}[Agentic RAG Answer]:{Colors.ENDC}")
    print(agentic_ans)

    # Print Judge Feedback
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'-' * 75}")
    print(f"LLM-as-a-Judge Evaluation Feedback:")
    print(f"{'-' * 75}{Colors.ENDC}")

    if has_gold:
        if not args.skip_vector:
            print(f"\n{Colors.BLUE}{Colors.BOLD}Vector RAG Judge Reasoning:{Colors.ENDC}")
            print(f"  {vector_reason}")
        print(f"\n{Colors.GREEN}{Colors.BOLD}Agentic RAG Judge Reasoning:{Colors.ENDC}")
        print(f"  {agentic_reason}")
    else:
        if not args.skip_vector:
            print(f"\n{Colors.BLUE}{Colors.BOLD}Vector RAG strengths/weaknesses:{Colors.ENDC}")
            print(f"  {vector_reason}")
            print(f"\n{Colors.GREEN}{Colors.BOLD}Agentic RAG strengths/weaknesses:{Colors.ENDC}")
            print(f"  {agentic_reason}")
            print(f"\n{Colors.HEADER}{Colors.BOLD}Judge Comparison Summary:{Colors.ENDC}")
            print(f"  {comp_summary}")
        else:
            print(f"  Comparison requires running both pipelines. Run without --skip-vector.")

    # 5. Write Persistent Markdown Report
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_file = results_dir / f"single_test_{timestamp}.md"

    md_content = []
    md_content.append(f"# RAG Head-to-Head Comparison Report")
    md_content.append(f"- **PDF Location:** `{pdf_path}`")
    md_content.append(f"- **Question:** {question}")
    if has_gold:
        md_content.append(f"- **Gold Answer:** *{gold_answer}*")
    md_content.append(f"- **Evaluation Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md_content.append(f"- **LLM & Judge Model:** `{args.model}`\n")

    md_content.append("## Metrics Summary\n")
    md_content.append("| RAG System | Latency (s) | Quality Score | Faithfulness |")
    md_content.append("| :--- | :--- | :--- | :--- |")
    md_content.append(f"| **Agentic RAG** | {agentic_time:.2f}s | {agentic_score}/5 | {agentic_faith_str} |")
    if not args.skip_vector:
        md_content.append(f"| **Vector RAG** | {vector_time:.2f}s | {vector_score}/5 | {vector_faith_str} |")
    md_content.append("\n")

    md_content.append("## Generated Answers\n")
    if not args.skip_vector:
        md_content.append("### Vector RAG Answer\n")
        md_content.append(f"```text\n{vector_ans}\n```\n")
    md_content.append("### Agentic RAG Answer\n")
    md_content.append(f"```text\n{agentic_ans}\n```\n")

    md_content.append("## Judge Detailed Feedback\n")
    if has_gold:
        if not args.skip_vector:
            md_content.append("### Vector RAG Judge Analysis\n")
            md_content.append(f"- **Correctness:** {vector_score}/5\n")
            md_content.append(f"- **Faithfulness:** {vector_faith}/5\n")
            md_content.append(f"- **Reasoning:** {vector_reason}\n\n")
        
        md_content.append("### Agentic RAG Judge Analysis\n")
        md_content.append(f"- **Correctness:** {agentic_score}/5\n")
        md_content.append(f"- **Faithfulness:** {agentic_faith}/5\n")
        md_content.append(f"- **Reasoning:** {agentic_reason}\n")
    else:
        if not args.skip_vector:
            md_content.append("### Vector RAG strengths/weaknesses\n")
            md_content.append(f"{vector_reason}\n\n")
            md_content.append("### Agentic RAG strengths/weaknesses\n")
            md_content.append(f"{agentic_reason}\n\n")
            md_content.append("### Judge Comparison Summary\n")
            md_content.append(f"{comp_summary}\n")

    report_file.write_text("\n".join(md_content), encoding="utf-8")
    print(f"\n{Colors.YELLOW}{Colors.BOLD}Detailed Markdown Report saved to:{Colors.ENDC} {report_file}\n")


if __name__ == "__main__":
    main()
