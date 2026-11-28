"""Bounded local response streaming; previews never expose tool proposals."""
from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable

import httpx

from companion.providers import ProviderResult

MAX_RESPONSE_BYTES = 512_000
MAX_TEXT = 40_000
CHAT_CONTEXT_TOKENS = 4096
CHAT_REPLY_TOKENS = 600
CHAT_KEEP_ALIVE = "2m"
RESOURCE_PROFILES = ("balanced", "efficient")


def resource_limits(profile: str = "balanced") -> dict:
    """Explicit per-request budgets, not OS CPU/memory utilization guarantees.

    Preserve the conversation window and answer length in both modes. Reducing
    threads/batches leaves more headroom for other apps; Quiet also disables GPU
    offload and releases the model sooner, at the cost of slower/cold replies.
    """
    if profile not in RESOURCE_PROFILES:
        raise ValueError("Choose Balanced or Quiet for local resource usage.")
    available = (getattr(os, "process_cpu_count", os.cpu_count)() or 1)
    threads = max(1, min(2 if profile == "efficient" else 4, available - 1))
    options = {"num_thread": threads, "num_batch": 128 if profile == "efficient" else 256}
    if profile == "efficient":
        options["num_gpu"] = 0
    return {"keep_alive": "30s" if profile == "efficient" else CHAT_KEEP_ALIVE,
            "options": options}


def preview_text(text: str, json_mode: bool) -> str:
    if not json_mode:
        return text[:12000]
    # Only render complete character/escape units inside the message string.
    # Proposal fields remain invisible and cannot dispatch an action.
    match = re.search(r'"message"\s*:\s*"((?:[^"\\]|\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4}))*)', text)
    if not match:
        return ""
    try:
        return json.loads('"' + match[1] + '"')[:12000].encode("utf-8", errors="replace").decode("utf-8")
    except (ValueError, TypeError):
        return ""


async def stream_chat(settings, *, system: str, messages: list[dict], json_mode: bool,
                      on_preview: Callable[[str], None], resource_profile: str = "balanced") -> ProviderResult:
    """Caller verifies loopback endpoint and downloaded-model metadata first."""
    # Let the installed model apply its own trained conversation template. XML
    # role labels inside /generate's user prompt are not actual system/history
    # messages, and small local models readily confuse those roles.
    limits = resource_limits(resource_profile)
    payload = {"model": settings.ollama_model,
               "messages": [{"role": "system", "content": system}, *messages], "stream": True,
               "think": False, "keep_alive": limits["keep_alive"],
               "options": {"temperature": .5, "num_predict": CHAT_REPLY_TOKENS,
                           "num_ctx": CHAT_CONTEXT_TOKENS, **limits["options"]}}
    if json_mode:
        payload["format"] = "json"
    text = ""
    last_preview = ""
    last_emit = 0.0
    done = False
    total = 0
    pending = bytearray()
    final = {}
    async with httpx.AsyncClient(timeout=settings.request_timeout_seconds, trust_env=False,
                                 follow_redirects=False) as client:
        async with client.stream("POST", settings.ollama_base_url.rstrip("/") + "/api/chat",
                                 json=payload, headers={"Accept-Encoding": "identity"}) as response:
            response.raise_for_status()
            if response.headers.get("content-encoding", "identity").lower() != "identity":
                raise ValueError("Unexpected compressed local response")
            # A fixed chunk_size makes httpx wait for a whole block before
            # yielding. Local token packets are often less than 200 bytes.
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if total > MAX_RESPONSE_BYTES:
                    raise ValueError("Local response exceeded its size limit")
                pending.extend(chunk)
                while b"\n" in pending:
                    line, _, remainder = pending.partition(b"\n")
                    pending = bytearray(remainder)
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if not isinstance(item, dict) or item.get("error"):
                        raise ValueError("The local engine stopped the response")
                    message = item.get("message", {})
                    if not isinstance(message, dict):
                        raise ValueError("Invalid local response message")
                    delta = message.get("content", "")
                    if not isinstance(delta, str) or len(text) + len(delta) > MAX_TEXT:
                        raise ValueError("Invalid local response text")
                    text += delta
                    preview = preview_text(text, json_mode)
                    now = time.monotonic()
                    if preview != last_preview and (now - last_emit >= .15 or item.get("done")):
                        on_preview(preview)
                        last_preview, last_emit = preview, now
                    if item.get("done") is True:
                        done, final = True, item
                if done:
                    break
    if not done or not text.strip():
        raise ValueError("The local response ended before completion")
    parsed = {}
    if json_mode:
        try:
            value = json.loads(text)
            if isinstance(value, dict):
                parsed = value
        except ValueError:
            pass
        if final.get("done_reason") == "length" or not isinstance(parsed.get("message"), str):
            partial = preview_text(text, True)
            if not partial:
                raise ValueError("The local response did not contain a usable answer")
            parsed = {"message": partial + "\n\nThis answer is incomplete. Try a shorter request.", "incomplete": True}
    elif final.get("done_reason") == "length":
        # Plain conversation has no JSON envelope to flag a bounded generation.
        # Preserve its useful text while making the limit visible in both the
        # last streaming preview and the stored response.
        text += "\n\nThis answer is incomplete. Ask me to continue."
        parsed = {"incomplete": True}
        preview = preview_text(text, False)
        if preview != last_preview:
            on_preview(preview)
    return ProviderResult(text=text, parsed=parsed, provider="ollama", model=settings.ollama_model,
        attempted=["ollama"], prompt_tokens=final.get("prompt_eval_count", 0),
        completion_tokens=final.get("eval_count", 0))
