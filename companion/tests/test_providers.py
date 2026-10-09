"""Provider chain tests — normalized output from each provider."""

from __future__ import annotations

import asyncio
import json

import httpx

from companion.config import BrainSettings
from companion.providers import GeminiProvider, OllamaProvider, ProviderChain


def _run(coro):
    return asyncio.run(coro)


def test_gemini_provider_parses_response(settings: BrainSettings) -> None:
    """Gemini must return a normalized result (regression: missing `text` kwarg)."""
    sb = settings.model_copy(update={"gemini_api_key": "test-key"})
    provider = GeminiProvider(sb)
    assert provider.available is True

    def handler(request: httpx.Request) -> httpx.Response:
        assert "generateContent" in str(request.url)
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {"content": {"parts": [{"text": '{"text": "pong", "tone": "warm"}'}]}}
                ],
                "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 5},
            },
        )

    transport = httpx.MockTransport(handler)

    async def scenario():
        async with httpx.AsyncClient(transport=transport) as client:
            return await provider.generate(
                system="test",
                messages=[{"role": "user", "content": "hi"}],
                json_mode=True,
                client=client,
            )

    result = _run(scenario())
    assert result.provider == "gemini"
    assert result.text == '{"text": "pong", "tone": "warm"}'
    assert result.parsed == {"text": "pong", "tone": "warm"}
    assert result.prompt_tokens == 12
    assert result.completion_tokens == 5


def test_gemini_uses_relai_model_override(settings: BrainSettings) -> None:
    """RELAI_MODEL must win over the companion_model default."""
    sb = settings.model_copy(update={"relai_model": "gemini-flash-latest"})
    assert sb.gemini_model == "gemini-flash-latest"
    sb2 = settings.model_copy(update={"relai_model": ""})
    assert sb2.gemini_model == sb2.companion_model


def test_chain_skips_unavailable_providers(settings: BrainSettings) -> None:
    """Chain reports an empty attempt list when nothing is configured."""
    chain = ProviderChain(settings)
    assert chain.health() == {
        "gemini": False,
        "openai": False,
        "ollama": True,  # default base url
        "anthropic": False,
    }


def test_local_inference_disables_proxies_redirects_and_thinking(settings, monkeypatch):
    """Use the configured CPU timeout without sending local prompts through proxies."""
    options = []
    sent = []
    original_client = httpx.AsyncClient

    def respond(request):
        sent.append(request)
        assert request.url == "http://127.0.0.1:11434/api/generate"
        payload = json.loads(request.content)
        assert payload["think"] is False
        assert payload["keep_alive"] == "1m"
        assert payload["stream"] is False
        assert payload["options"]["num_predict"] == 600
        assert payload["model"] == "downloaded:3b"
        assert "A private local question" in payload["prompt"]
        return httpx.Response(200, json={"response": "A local answer", "model": "downloaded:3b", "prompt_eval_count": 20, "eval_count": 3})

    def client(**kwargs):
        options.append(kwargs)
        return original_client(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    provider = OllamaProvider(settings.model_copy(update={
        "ollama_base_url": "http://127.0.0.1:11434", "ollama_model": "downloaded:3b", "request_timeout_seconds": 180,
    }))
    result = _run(provider.generate(system="test", messages=[{"role": "user", "content": "A private local question"}], max_tokens=600, json_mode=False))
    assert len(sent) == 1
    assert options == [{"timeout": 180, "trust_env": False, "follow_redirects": False}]
    assert result.text == "A local answer"
    assert result.prompt_tokens == 20
    assert result.completion_tokens == 3
