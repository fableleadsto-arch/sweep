"""Local text generation for tutorial-style answers — no external APIs.

Resolution order:
  1. Ollama (local server) when running with a model installed.
  2. A cached transformers model (Qwen2-0.5B-Instruct) via torch on CPU.
  3. A rule-shaped fallback that repeats intelligible structure so the
     terminal ALWAYS has a reply, even fully offline with no model.

The loaded model is cached per process behind a lock so parallel callers
(e.g. the dataset evaluator) never double-load it.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Optional

_LOCK = threading.Lock()
_CACHED: dict[str, Any] = {}


def available_backends() -> list[str]:
    out = []
    if _ollama_ping():
        out.append("ollama")
    if _qwen_available():
        out.append("transformers (Qwen2-0.5B-Instruct)")
    out.append("rule fallback")
    return out


def generate(prompt: str, max_tokens: int = 300, temperature: float = 0.7, system: str = "") -> str:
    """Generate a reply using the best local backend available."""
    try:
        if _ollama_ping():
            reply = _ollama_generate(prompt, max_tokens=max_tokens, temperature=temperature, system=system)
            if reply:
                return reply.strip()
    except Exception:  # noqa: BLE001 — fall through to next backend
        pass
    try:
        if _qwen_available():
            reply = _qwen_generate(prompt, max_tokens=max_tokens, temperature=temperature, system=system)
            if reply:
                return reply.strip()
    except Exception:  # noqa: BLE001
        pass
    return _rule_reply(prompt, system)


# ── ollama ────────────────────────────────────────────────────────────


def _ollama_base() -> str:
    return os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")


def _ollama_ping() -> bool:
    if "OLLAMA_SERVER" in os.environ and os.environ["OLLAMA_SERVER"].lower() in {"0", "false", "off"}:
        return False
    try:
        import httpx

        r = httpx.get(_ollama_base() + "/api/tags", timeout=2.0)
        data = r.json()
        return bool(data.get("models"))
    except Exception:  # noqa: BLE001
        return False


def _ollama_models() -> list[str]:
    try:
        import httpx

        r = httpx.get(_ollama_base() + "/api/tags", timeout=2.0)
        return [str(m["name"]) for m in r.json().get("models", [])]
    except Exception:  # noqa: BLE001
        return []


def _pick_ollama_model() -> Optional[str]:
    models = _ollama_models()
    if not models:
        return None
    for preferred in ("qwen2.5-coder:3b", "qwen2.5-coder:7b", "llama3.2:3b", "tinyllama", "phi3"):
        for m in models:
            if m.startswith(preferred):
                return m
    return models[0]


def _ollama_generate(prompt: str, max_tokens: int, temperature: float, system: str) -> str:
    import httpx

    model = _pick_ollama_model()
    if not model:
        return ""
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": temperature},
    }
    if system:
        body["system"] = system
    r = httpx.post(_ollama_base() + "/api/generate", json=body, timeout=180.0)
    r.raise_for_status()
    return str(r.json().get("response") or "")


# ── transformers (cached Qwen2-0.5B-Instruct) ─────────────────────────


def _qwen_available() -> bool:
    from pathlib import Path

    hub = Path.home() / ".cache" / "huggingface" / "hub"
    return (hub / "models--Qwen--Qwen2-0.5B-Instruct").is_dir()


def _load_qwen():
    """Load once, guarded by a lock (model lives in _CACHED["qwen"])."""
    if "qwen" not in _CACHED:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        name = "Qwen/Qwen2-0.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(name, local_files_only=True)
        model.eval()
        _CACHED["qwen"] = (model, tokenizer)
    return _CACHED["qwen"]


def _qwen_generate(prompt: str, max_tokens: int, temperature: float, system: str) -> str:
    import torch

    with _LOCK:
        model, tokenizer = _load_qwen()
    if system:
        prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    else:
        prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip()


# ── rule fallback ─────────────────────────────────────────────────────


def _rule_reply(prompt: str, system: str) -> str:
    """Last-resort reply so discussion never silently fails."""
    head = "I'm running with no local model available right now."
    tail = (
        "\n\nTry: 'search for <topic>' to look things up, 'data list' to explore "
        "datasets, or 'help' for everything I can do."
    )
    if system:
        return f"{head} {system}{tail}"
    snippet = " ".join(prompt.split()[:24])
    return f"{head} I'll paraphrase your question: \"{snippet}\".{tail}"