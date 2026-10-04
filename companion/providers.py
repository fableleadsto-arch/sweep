"""Multi-provider LLM gateway — mirrors the TypeScript `gateway` chain.

Order (matching `brain.py` / `gateway.server.ts`): Gemini → OpenAI → Ollama
(local) → Anthropic. Every provider returns a normalized `ProviderResult`
shape so callers never deal with vendor-specific response formats.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from .config import BrainSettings


@dataclass
class ProviderResult:
    """Normalized output from any LLM provider."""

    text: str
    parsed: dict[str, Any] = field(default_factory=dict)
    model: str = ""
    provider: str = ""
    attempted: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


class BaseProvider:
    """Base class for all LLM providers."""

    name: str = "base"
    settings: BrainSettings

    def __init__(self, settings: BrainSettings) -> None:
        self.settings = settings

    @property
    def available(self) -> bool:
        return False

    async def generate(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.5,
        max_tokens: int = 1200,
        json_mode: bool = True,
        client: Optional[httpx.AsyncClient] = None,
    ) -> ProviderResult:
        raise NotImplementedError

    def _parse_json(self, text: str) -> dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            start, end = cleaned.find("{"), cleaned.rfind("}")
            if start != -1 and end != -1:
                try:
                    parsed = json.loads(cleaned[start : end + 1])
                    return parsed if isinstance(parsed, dict) else {}
                except json.JSONDecodeError:
                    pass
        return {}


class GeminiProvider(BaseProvider):
    """Google Gemini — the primary brain provider."""

    name = "gemini"

    @property
    def available(self) -> bool:
        return bool(self.settings.gemini_api_key)

    async def generate(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.5,
        max_tokens: int = 1200,
        json_mode: bool = True,
        client: Optional[httpx.AsyncClient] = None,
    ) -> ProviderResult:
        model = self.settings.gemini_model
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
            ":generateContent"
        )
        payload: dict[str, Any] = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [
                {"role": msg.get("role", "user"), "parts": [{"text": msg.get("content", "")}]}
                for msg in messages
            ],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        if json_mode:
            payload["generationConfig"]["responseMimeType"] = "application/json"

        own_client = client is None
        client = client or httpx.AsyncClient(timeout=self.settings.request_timeout_seconds)
        try:
            resp = await client.post(url, params={"key": self.settings.gemini_api_key}, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Gemini request failed: {exc}") from exc
        finally:
            if own_client:
                await client.aclose()

        result = ProviderResult(text="", model=model, provider=self.name)
        try:
            candidate = data["candidates"][0]
            result.text = candidate["content"]["parts"][0]["text"]
            if json_mode:
                result.parsed = self._parse_json(result.text)
        except (KeyError, IndexError):
            pass
        usage = data.get("usageMetadata", {})
        result.prompt_tokens = usage.get("promptTokenCount", 0)
        result.completion_tokens = usage.get("candidatesTokenCount", 0)
        return result


class OpenAIProvider(BaseProvider):
    """OpenAI chat completions — secondary fallback."""

    name = "openai"

    @property
    def available(self) -> bool:
        return bool(self.settings.openai_api_key)

    async def generate(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.5,
        max_tokens: int = 1200,
        json_mode: bool = True,
        client: Optional[httpx.AsyncClient] = None,
    ) -> ProviderResult:
        api_messages = [{"role": "system", "content": system}]
        for msg in messages:
            role = "assistant" if msg.get("role") == "model" else msg.get("role", "user")
            api_messages.append({"role": role, "content": msg.get("content", "")})

        payload: dict[str, Any] = {
            "model": self.settings.openai_model,
            "messages": api_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        own_client = client is None
        client = client or httpx.AsyncClient(timeout=self.settings.request_timeout_seconds)
        try:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.settings.openai_api_key}"},
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"OpenAI request failed: {exc}") from exc
        finally:
            if own_client:
                await client.aclose()

        text = data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        result = ProviderResult(
            text=text,
            parsed=self._parse_json(text) if json_mode and text else {},
            model=data.get("model", self.settings.openai_model),
            provider=self.name,
        )
        usage = data.get("usage", {})
        result.prompt_tokens = usage.get("prompt_tokens", 0)
        result.completion_tokens = usage.get("completion_tokens", 0)
        return result


class OllamaProvider(BaseProvider):
    """Local Ollama inference — free, self-hosted fallback."""

    name = "ollama"

    @property
    def available(self) -> bool:
        return bool(self.settings.ollama_base_url)

    async def generate(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.5,
        max_tokens: int = 1200,
        json_mode: bool = True,
        client: Optional[httpx.AsyncClient] = None,
    ) -> ProviderResult:
        prompt_parts = [f"<system>\n{system}\n</system>"]
        for msg in messages:
            prompt_parts.append(f"<{msg.get('role', 'user')}>\n{msg.get('content', '')}\n</{msg.get('role', 'user')}>")
        payload: dict[str, Any] = {
            "model": self.settings.ollama_model,
            "prompt": "\n".join(prompt_parts),
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if json_mode:
            payload["format"] = "json"

        own_client = client is None
        client = client or httpx.AsyncClient(timeout=self.settings.request_timeout_seconds)
        try:
            resp = await client.post(
                f"{self.settings.ollama_base_url}/api/generate", json=payload
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc
        finally:
            if own_client:
                await client.aclose()

        text = data.get("response", "")
        result = ProviderResult(
            text=text,
            parsed=self._parse_json(text) if json_mode else {},
            model=data.get("model", self.settings.ollama_model),
            provider=self.name,
        )
        result.prompt_tokens = data.get("prompt_eval_count", 0)
        result.completion_tokens = data.get("eval_count", 0)
        return result


class AnthropicProvider(BaseProvider):
    """Anthropic Claude — tertiary fallback."""

    name = "anthropic"

    @property
    def available(self) -> bool:
        return bool(self.settings.anthropic_api_key)

    async def generate(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.5,
        max_tokens: int = 1200,
        json_mode: bool = True,
        client: Optional[httpx.AsyncClient] = None,
    ) -> ProviderResult:
        api_messages = []
        for msg in messages:
            role = "assistant" if msg.get("role") == "model" else msg.get("role", "user")
            api_messages.append({"role": role, "content": msg.get("content", "")})

        payload: dict[str, Any] = {
            "model": self.settings.anthropic_model,
            "system": system,
            "messages": api_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "x-api-key": self.settings.anthropic_api_key,
            "anthropic-version": "2023-06-01",
        }

        own_client = client is None
        client = client or httpx.AsyncClient(timeout=self.settings.request_timeout_seconds)
        try:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages", headers=headers, json=payload
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Anthropic request failed: {exc}") from exc
        finally:
            if own_client:
                await client.aclose()

        text = "".join(
            block.get("text", "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        )
        result = ProviderResult(
            text=text,
            parsed=self._parse_json(text) if json_mode and text else {},
            model=data.get("model", self.settings.anthropic_model),
            provider=self.name,
        )
        usage = data.get("usage", {})
        result.prompt_tokens = usage.get("input_tokens", 0)
        result.completion_tokens = usage.get("output_tokens", 0)
        return result


class ProviderChain:
    """Tries providers in configured order; returns the first success."""

    def __init__(self, settings: BrainSettings) -> None:
        self.settings = settings
        registry: dict[str, type[BaseProvider]] = {
            "gemini": GeminiProvider,
            "openai": OpenAIProvider,
            "ollama": OllamaProvider,
            "anthropic": AnthropicProvider,
        }
        self.providers: dict[str, BaseProvider] = {
            name: registry[name](settings) for name in registry
        }

    def health(self) -> dict[str, bool]:
        return {name: p.available for name, p in self.providers.items()}

    def _ordered(self, preferred: Optional[str]) -> list[BaseProvider]:
        order = [p for p in self.settings.provider_order if p in self.providers]
        if preferred and preferred in self.providers and preferred not in order:
            order.insert(0, preferred)
        if preferred in order:
            order.remove(preferred)
            order.insert(0, preferred)
        available = [self.providers[p] for p in order if self.providers[p].available]
        return available or [self.providers[p] for p in order]

    async def generate(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.5,
        max_tokens: int = 1200,
        json_mode: bool = True,
        preferred: Optional[str] = None,
    ) -> ProviderResult:
        attempted: list[str] = []
        last_error: Optional[str] = None
        for provider in self._ordered(preferred):
            attempted.append(provider.name)
            try:
                result = await provider.generate(
                    system=system,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                )
                if result.text.strip():
                    result.attempted = attempted
                    return result
            except Exception as exc:  # noqa: BLE001 - fall through the chain
                last_error = str(exc)
                continue

        raise RuntimeError(
            f"All AI providers failed. Tried: {', '.join(attempted) or 'none'}. "
            f"Last error: {last_error or 'no provider available'}"
        )
