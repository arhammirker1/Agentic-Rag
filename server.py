"""
server.py — FastAPI backend for PageIndex Web UI.

Provides notebook management, PDF upload, and RAG-powered chat.
Logs are written to files only — no console output in web mode.

Usage:
    python server.py
    # Open http://localhost:8000 in your browser
"""

import io
import json
import os
import sys
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agenticrag import Forest, ForestConfig, GroqModel
from agenticrag.config import LocalModel

# ── Constants ─────────────────────────────────────────────────────────────

DATA_DIR = Path("notebooks_data")
NOTEBOOKS_FILE = DATA_DIR / "notebooks.json"
SETTINGS_FILE = DATA_DIR / "settings.json"

DEFAULT_SETTINGS = {
    "provider": "groq",           # "groq" or "local"
    "model": GroqModel.LLAMA4_SCOUT,
    "api_key": "",                # read from env if empty
    "base_url": "",               # for local LLMs
}

# ── App ───────────────────────────────────────────────────────────────────

app = FastAPI(title="PageIndex", docs_url="/docs")

# In-memory caches
_forests: dict = {}       # notebook_id -> Forest instance
_chat_history: dict = {}  # notebook_id -> list of messages


# ── Helpers ───────────────────────────────────────────────────────────────

def _ensure_dirs():
    DATA_DIR.mkdir(exist_ok=True)
    if not NOTEBOOKS_FILE.exists():
        NOTEBOOKS_FILE.write_text("[]", encoding="utf-8")
    if not SETTINGS_FILE.exists():
        SETTINGS_FILE.write_text(json.dumps(DEFAULT_SETTINGS, indent=2), encoding="utf-8")


def _load_notebooks() -> list:
    _ensure_dirs()
    return json.loads(NOTEBOOKS_FILE.read_text(encoding="utf-8"))


def _save_notebooks(notebooks: list):
    NOTEBOOKS_FILE.write_text(json.dumps(notebooks, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_settings() -> dict:
    _ensure_dirs()
    return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))


def _save_settings(settings: dict):
    SETTINGS_FILE.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def _get_notebook(notebook_id: str) -> dict:
    for nb in _load_notebooks():
        if nb["id"] == notebook_id:
            return nb
    raise HTTPException(status_code=404, detail="Notebook not found")


def _nb_dir(notebook_id: str) -> Path:
    return DATA_DIR / notebook_id


def _sources_dir(notebook_id: str) -> Path:
    d = _nb_dir(notebook_id) / "sources"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get_forest(notebook_id: str) -> Forest:
    """Get or create a Forest instance for a notebook."""
    if notebook_id in _forests:
        return _forests[notebook_id]

    settings = _load_settings()
    forest_data = str(_nb_dir(notebook_id) / "forest_data")

    config_kwargs = {
        "model": settings.get("model", GroqModel.LLAMA4_SCOUT),
        "verbose": True,
        "quiet": False, # Show progress in the console
        "data_dir": forest_data,
        "enable_thinking": settings.get("enable_thinking", False),
        "num_ctx": settings.get("num_ctx", 32768),
    }

    if settings.get("provider") == "local" and settings.get("base_url"):
        config_kwargs["base_url"] = settings["base_url"]
        if settings.get("api_key"):
            config_kwargs["api_key"] = settings["api_key"]
        else:
            config_kwargs["api_key"] = "local"
    else:
        api_key = settings.get("api_key") or os.environ.get("GROQ_API_KEY", "")
        if api_key:
            config_kwargs["api_key"] = api_key

    forest = Forest(**config_kwargs)
    _forests[notebook_id] = forest
    return forest


# ── Models ────────────────────────────────────────────────────────────────

class CreateNotebook(BaseModel):
    name: str

class UpdateSettings(BaseModel):
    provider: str = "groq"
    model: str = GroqModel.LLAMA4_SCOUT
    api_key: str = ""
    base_url: str = ""
    enable_thinking: bool = False

class ChatRequest(BaseModel):
    message: str


# ── Routes: UI ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = Path(__file__).parent / "web" / "index.html"
    if not html_path.exists():
        raise HTTPException(500, "Frontend not found")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


# ── Routes: Notebooks ────────────────────────────────────────────────────

@app.get("/api/notebooks")
async def list_notebooks():
    return _load_notebooks()


@app.post("/api/notebooks")
async def create_notebook(body: CreateNotebook):
    notebooks = _load_notebooks()
    nb = {
        "id": str(uuid.uuid4())[:12],
        "name": body.name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sources": [],
        "indexed": False,
    }
    notebooks.append(nb)
    _save_notebooks(notebooks)
    _sources_dir(nb["id"])  # create directory
    return nb


@app.delete("/api/notebooks/{notebook_id}")
async def delete_notebook(notebook_id: str):
    notebooks = _load_notebooks()
    notebooks = [nb for nb in notebooks if nb["id"] != notebook_id]
    _save_notebooks(notebooks)
    # Clean up files
    nb_path = _nb_dir(notebook_id)
    if nb_path.exists():
        shutil.rmtree(nb_path, ignore_errors=True)
    # Clean up in-memory
    _forests.pop(notebook_id, None)
    _chat_history.pop(notebook_id, None)
    return {"ok": True}


# ── Routes: Sources ──────────────────────────────────────────────────────

@app.get("/api/notebooks/{notebook_id}/sources")
async def list_sources(notebook_id: str):
    nb = _get_notebook(notebook_id)
    return nb.get("sources", [])


@app.post("/api/notebooks/{notebook_id}/sources")
async def upload_source(notebook_id: str, file: UploadFile = File(...)):
    nb = _get_notebook(notebook_id)

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    # Save file
    dest = _sources_dir(notebook_id) / file.filename
    content = await file.read()
    dest.write_bytes(content)

    # Update notebook
    notebooks = _load_notebooks()
    for n in notebooks:
        if n["id"] == notebook_id:
            if file.filename not in [s["name"] for s in n.get("sources", [])]:
                n["sources"].append({
                    "name": file.filename,
                    "size": len(content),
                    "added_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                })
            n["indexed"] = False  # mark for re-indexing
            break
    _save_notebooks(notebooks)

    # Invalidate cached forest
    _forests.pop(notebook_id, None)

    return {"ok": True, "filename": file.filename, "size": len(content)}


@app.delete("/api/notebooks/{notebook_id}/sources/{filename}")
async def delete_source(notebook_id: str, filename: str):
    # Remove file
    f = _sources_dir(notebook_id) / filename
    if f.exists():
        f.unlink()

    # Update notebook
    notebooks = _load_notebooks()
    for n in notebooks:
        if n["id"] == notebook_id:
            n["sources"] = [s for s in n.get("sources", []) if s["name"] != filename]
            n["indexed"] = False
            break
    _save_notebooks(notebooks)
    _forests.pop(notebook_id, None)
    return {"ok": True}


# ── Routes: Chat ─────────────────────────────────────────────────────────

@app.post("/api/notebooks/{notebook_id}/chat")
async def chat_endpoint(notebook_id: str, body: ChatRequest):
    nb = _get_notebook(notebook_id)
    sources = nb.get("sources", [])

    if not sources:
        raise HTTPException(400, "No sources uploaded. Add PDFs first.")

    forest = _get_forest(notebook_id)

    # Index any un-indexed sources
    sources_path = _sources_dir(notebook_id)
    existing_docs = set()
    try:
        info = forest.info()
        if isinstance(info, dict):
            existing_docs = set()  # we'll re-index if needed
    except Exception:
        pass

    if not nb.get("indexed", False):
        # Index all PDFs in sources directory
        pdf_files = list(sources_path.glob("*.pdf"))
        if pdf_files:
            print(f"\n[server] 📂 Found {len(pdf_files)} PDF(s). Checking indices...", flush=True)
            
        for pdf in pdf_files:
            try:
                print(f"[server] ⚙️  Indexing '{pdf.name}'... (this might take a minute)", flush=True)
                forest.add(str(pdf))
            except Exception as e:
                print(f"[server] ❌ Error indexing {pdf.name}: {e}", flush=True)

        if pdf_files:
            print("[server] ✅ All sources are now indexed.\n", flush=True)

        # Mark as indexed
        notebooks = _load_notebooks()
        for n in notebooks:
            if n["id"] == notebook_id:
                n["indexed"] = True
                break
        _save_notebooks(notebooks)

    # Get/create chat history
    if notebook_id not in _chat_history:
        _chat_history[notebook_id] = []
    history = _chat_history[notebook_id]

    # Ask the question
    try:
        result = forest.ask(body.message, history=history)

        # Update history
        history.append({"role": "user", "content": body.message})
        history.append({"role": "assistant", "content": result.text})

        # Keep last 20 messages
        if len(history) > 20:
            _chat_history[notebook_id] = history[-20:]

        return {
            "answer": result.text,
            "confidence": result.confidence,
            "sources": result.sources,
            "documents_searched": result.documents_searched,
            "was_rewritten": result.was_rewritten,
            "elapsed": result.elapsed_seconds,
            "reasoning_trace": result.reasoning_trace,
            "hallucinations": result.hallucinations,
        }
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")


# ── Routes: Settings ─────────────────────────────────────────────────────

@app.get("/api/settings")
async def get_settings():
    settings = _load_settings()
    # Mask API key
    if settings.get("api_key"):
        settings["api_key_set"] = True
        settings["api_key"] = ""
    else:
        settings["api_key_set"] = bool(os.environ.get("GROQ_API_KEY"))
    return settings


@app.post("/api/settings")
async def update_settings(body: UpdateSettings):
    settings = {
        "provider": body.provider,
        "model": body.model,
        "api_key": body.api_key,
        "base_url": body.base_url,
        "enable_thinking": body.enable_thinking,
    }
    _save_settings(settings)
    # Invalidate all cached forests
    _forests.clear()
    return {"ok": True}


@app.get("/api/models")
async def list_models():
    """Return available model presets."""
    return {
        "groq": [
            {"id": GroqModel.LLAMA4_SCOUT, "name": "Llama 4 Scout (fast)"},
            {"id": GroqModel.LLAMA4_MAVERICK, "name": "Llama 4 Maverick"},
            {"id": GroqModel.GPT_OSS_20B, "name": "GPT-OSS 20B"},
            {"id": GroqModel.GPT_OSS_120B, "name": "GPT-OSS 120B (best)"},
            {"id": GroqModel.LLAMA3_3_70B, "name": "Llama 3.3 70B"},
            {"id": GroqModel.QWEN3_32B, "name": "Qwen 3 32B"},
            {"id": GroqModel.DEEPSEEK_R1_DISTILL_LLAMA_70B, "name": "DeepSeek R1 70B"},
        ],
        "local": [
            {"id": LocalModel.QWEN3_4B, "name": "Qwen 3 4B (2.5GB, 256K ctx) — recommended"},
            {"id": LocalModel.QWEN3_8B, "name": "Qwen 3 8B (5.2GB, 40K ctx)"},
            {"id": LocalModel.QWEN3_14B, "name": "Qwen 3 14B (9.3GB, 40K ctx)"},
            {"id": LocalModel.QWEN3_30B, "name": "Qwen 3 30B (19GB, 256K ctx)"},
            {"id": LocalModel.LLAMA3_2_3B, "name": "Llama 3.2 3B (2.0GB)"},
            {"id": LocalModel.MISTRAL, "name": "Mistral (4.1GB, 32K ctx)"},
            {"id": LocalModel.PHI4, "name": "Phi-4 (9.1GB, 16K ctx)"},
            {"id": LocalModel.GEMMA3_12B, "name": "Gemma 3 12B (8.1GB, 128K ctx)"},
        ]
    }


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import socket
    import uvicorn

    parser = argparse.ArgumentParser(description="PageIndex Web Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0 for LAN access)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    args = parser.parse_args()

    _ensure_dirs()

    # Get LAN IP for display
    def _get_lan_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "<your-ip>"

    lan_ip = _get_lan_ip()

    print("\n" + "="*60)
    print("  PageIndex Web UI — Booting Up...")
    print("="*60)

    print(f"  [1/3] Loading notebooks from {DATA_DIR}...", end="", flush=True)
    _load_notebooks()
    print(" Done.")

    print(f"  [2/3] Checking settings...", end="", flush=True)
    _load_settings()
    print(" Done.")

    print(f"  [3/3] Starting web server...", flush=True)
    print("")
    print("  PageIndex is Ready!")
    print("  ──────────────────────────────────────────────────────────")
    print(f"  Local:   http://localhost:{args.port}")
    print(f"  LAN:     http://{lan_ip}:{args.port}")
    print("  ──────────────────────────────────────────────────────────")
    print(f"  Share the LAN URL with devices on your network.")
    print(f"  Only devices on your local network can access it.\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
