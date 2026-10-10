"""Bounded local response streaming; previews never expose tool proposals."""
from __future__ import annotations

import json
import re
import time
from collections.abc import Callable

import httpx

from companion.providers import ProviderResult

MAX_RESPONSE_BYTES = 512_000
MAX_TEXT = 40_000


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
                      on_preview: Callable[[str], None]) -> ProviderResult:
    """Caller verifies loopback endpoint and downloaded-model metadata first."""
    prompt = [f"<system>\n{system}\n</system>"]
    prompt.extend(f"<{item['role']}>\n{item['content']}\n</{item['role']}>" for item in messages)
    payload = {"model": settings.ollama_model, "prompt": "\n".join(prompt), "stream": True,
               "think": False, "keep_alive": "1m", "options": {"temperature": .5, "num_predict": 600}}
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
        async with client.stream("POST", settings.ollama_base_url.rstrip("/") + "/api/generate",
                                 json=payload, headers={"Accept-Encoding": "identity"}) as response:
            response.raise_for_status()
            if response.headers.get("content-encoding", "identity").lower() != "identity":
                raise ValueError("Unexpected compressed local response")
            async for chunk in response.aiter_bytes(chunk_size=1024):
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
                    delta = item.get("response", "")
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
    return ProviderResult(text=text, parsed=parsed, provider="ollama", model=settings.ollama_model,
        attempted=["ollama"], prompt_tokens=final.get("prompt_eval_count", 0),
        completion_tokens=final.get("eval_count", 0))
