"""
groq_client.py — LLM wrapper for PageIndex.

Supports two backends:
  1. Groq SDK (default) — uses GROQ_API_KEY
  2. Any OpenAI-compatible endpoint — set base_url in config
     Works with: Ollama, LM Studio, vLLM, llama.cpp, etc.

Thinking mode:
  Models like Qwen3 support a /think mode for deeper reasoning.
  Controlled via enable_thinking:
    - False (default) = fast mode, appends /no_think to prompts
    - True            = deep thinking, slower but higher quality
  <think> tags are always stripped from output regardless of setting.

Context window:
  For local LLMs, we pass num_ctx to Ollama to expand the context
  window. With 32GB+ RAM, the KV cache overflows from VRAM into
  system RAM automatically — no evidence truncation needed.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

_groq_client = None   # module-level singleton (Groq SDK)
_local_client = None  # module-level singleton (httpx for local LLMs)
_gemini_client = None # module-level singleton (Google GenAI)

# Default context window for local LLMs.
# Ollama's default is 2048 which is far too small for RAG evidence.
# 32768 tokens uses ~2-4GB of RAM for the KV cache on a 4B model.
# If VRAM is insufficient, Ollama automatically spills to system RAM.
LOCAL_NUM_CTX = 32768


def _get_client(api_key: Optional[str] = None, base_url: Optional[str] = None):
    """
    Return a cached LLM client.

    If base_url is set  → returns (httpx_client, base_url) for local LLMs
    Otherwise           → returns (groq_client, None) for Groq
    """
    if base_url:
        global _local_client
        if _local_client is None:
            import httpx
            # Local LLMs need generous timeouts:
            #   - First request loads model into VRAM (30-60s)
            #   - Generation on small GPUs can take 5-10 min for complex prompts
            #   - Multi-agent pipeline = many sequential calls
            _local_client = httpx.Client(timeout=httpx.Timeout(
                connect=60.0,
                read=3600.0,   # 1 hour — prevents timeout on long generations
                write=60.0,
                pool=60.0,
            ))
        return _local_client, base_url

    global _groq_client
    if _groq_client is None or api_key:
        try:
            from groq import Groq  # type: ignore
        except ImportError:
            raise ImportError(
                "The `groq` package is required.\n"
                "Install it with:  pip install groq\n"
                "Or:               pip install pageindex"
            )
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise ValueError(
                "No Groq API key found.\n"
                "Either pass api_key=... to PageIndexConfig, or set the "
                "GROQ_API_KEY environment variable.\n"
                "Get a free key at: https://console.groq.com"
            )
        _groq_client = Groq(api_key=key)
    return _groq_client, None

def _get_gemini_client(api_key: Optional[str] = None):
    """Return a cached Google GenAI client."""
    global _gemini_client
    if _gemini_client is None or api_key:
        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "The `google-genai` package is required to use Gemini.\n"
                "Install it with:  pip install google-genai"
            )
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "No Gemini API key found.\n"
                "Either pass api_key=... to PageIndexConfig, or set the "
                "GEMINI_API_KEY environment variable.\n"
                "Get a free key at: https://aistudio.google.com/app/apikey"
            )
        _gemini_client = genai.Client(api_key=key)
    return _gemini_client


def chat(
    prompt: str,
    *,
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    system: str = "",
    temperature: float = 0.0,
    max_tokens: int = 4096,
    quiet: bool = False,
    enable_thinking: bool = False,
    num_ctx: int = 32768,
) -> str:
    """
    Call an LLM chat completions endpoint and return the response text.

    Parameters
    ----------
    prompt           : User message content
    model            : Model ID  (e.g. "meta-llama/llama-4-scout" or "qwen3:4b")
    api_key          : API key (falls back to GROQ_API_KEY env var)
    base_url         : Custom endpoint for local LLMs
    system           : Optional system message
    temperature      : Sampling temperature
    max_tokens       : Max tokens in the response
    quiet            : If True, suppress console logging
    enable_thinking  : If True, allow Qwen3/DeepSeek deep thinking mode
                       If False (default), append /no_think for faster responses
    """
    is_gemini = model.lower().startswith("gemini")
    
    if is_gemini:
        gemini_client = _get_gemini_client(api_key)
        client, local_url = None, None
    else:
        gemini_client = None
        client, local_url = _get_client(api_key, base_url)

    # Build messages
    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})

    # Thinking mode control for local LLMs
    user_content = prompt
    if local_url and not enable_thinking:
        user_content = prompt.rstrip() + " /no_think"

    messages.append({"role": "user", "content": user_content})

    # --- LOGGING ---
    from .utils.logging import trail
    trail.step(f"LLM REQUEST ({model})", f"Max Tokens: {max_tokens} | Temp: {temperature} | Thinking: {enable_thinking}", {
        "system": system,
        "prompt": prompt
    }, quiet=quiet)

    if local_url:
        # ── Local LLM via OpenAI-compatible API ──
        url = local_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            # Tell Ollama to use a large context window.
            # The KV cache spills from VRAM → system RAM automatically,
            # so this works even on GPUs with limited VRAM (e.g. 5GB).
            # This key is ignored by non-Ollama backends.
            "options": {
                "num_ctx": num_ctx,
            },
        }
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        output = data["choices"][0]["message"]["content"] or ""
    elif is_gemini:
        # ── Google GenAI SDK ──
        from google import genai
        config_kwargs = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if system:
            config_kwargs["system_instruction"] = system
            
        response = gemini_client.models.generate_content(
            model=model,
            contents=user_content,
            config=genai.types.GenerateContentConfig(**config_kwargs)
        )
        output = response.text or ""
    else:
        # ── Groq SDK ──
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        output = response.choices[0].message.content or ""

    # Always strip <think> tags — they break JSON parsing and pollute answers
    output = _strip_thinking(output)

    trail.step(f"LLM RESPONSE ({model})", "Raw output received:", output, quiet=quiet)
    return output


def chat_json(
    prompt: str,
    *,
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    system: str = "",
    temperature: float = 0.0,
    max_tokens: int = 4096,
    quiet: bool = False,
    enable_thinking: bool = False,
    num_ctx: int = 32768,
) -> Any:
    """
    Like chat() but automatically parses the response as JSON.
    Handles markdown code fences (```json ... ```) gracefully.
    """
    json_reminder = "\n\nRemember: respond with valid JSON only. No prose, no markdown fences."
    raw = chat(
        prompt + json_reminder,
        model=model,
        api_key=api_key,
        base_url=base_url,
        system=system,
        temperature=temperature,
        max_tokens=max_tokens,
        quiet=quiet,
        enable_thinking=enable_thinking,
        num_ctx=num_ctx,
    )
    return _parse_json(raw)


# ── Internal ──────────────────────────────────────────────────────────────

def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from model output.

    Always applied regardless of enable_thinking setting, because
    even when thinking is enabled, the tags must be stripped before
    the output is used for JSON parsing or answer display.
    """
    if "<think>" not in text:
        return text
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return cleaned if cleaned else text


def _parse_json(text: str) -> Any:
    """Strip optional markdown fences and parse JSON."""
    text = text.strip()
    # Remove ```json or ``` wrappers
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text)
    return json.loads(text.strip())