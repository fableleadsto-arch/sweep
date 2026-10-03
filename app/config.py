"""Application configuration — every tunable lives here, driven by environment variables."""

from __future__ import annotations

from functools import lru_cache
from pydantic import Field

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed, validated configuration for the Sweep application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── App ────────────────────────────────────────────────────────
    app_name: str = "Sweep"
    debug: bool = Field(default=False, validation_alias="SWEEP_DEBUG")
    host: str = "127.0.0.1"
    port: int = 8787
    sweep_api_token: str = ""

    # ── CORS ───────────────────────────────────────────────────────
    cors_origins: str = ""

    # ── Search providers ───────────────────────────────────────────
    tavily_api_key: str = ""
    exa_api_key: str = ""
    searxng_base_url: str = ""
    jina_api_key: str = ""

    # ── Browser automation ─────────────────────────────────────────
    playwright_ws_endpoint: str = ""
    browser_ws_endpoint: str = ""
    # Remote browsers require their own network isolation for redirects/subresources.
    allow_remote_browser: bool = False

    # ── HTTP fetch ─────────────────────────────────────────────────
    relai_proxy_url: str = ""
    fetch_timeout_ms: int = 15_000
    fetch_max_retries: int = 2
    fetch_max_concurrent: int = 6

    # ── AI providers ───────────────────────────────────────────────
    gemini_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    # ── Database ───────────────────────────────────────────────────
    supabase_url: str = ""
    supabase_key: str = ""

    # ── Auth ───────────────────────────────────────────────────────
    service_token: str = ""

    # ── Derived helpers ────────────────────────────────────────────
    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def playwright_configured(self) -> bool:
        return bool(self.playwright_ws_endpoint or self.browser_ws_endpoint)

    @property
    def has_ai_provider(self) -> bool:
        return bool(self.gemini_api_key or self.openai_api_key or self.anthropic_api_key)

    @property
    def search_providers_configured(self) -> list[str]:
        providers = ["keyless"]  # Always available
        if self.tavily_api_key:
            providers.append("tavily")
        if self.exa_api_key:
            providers.append("exa")
        if self.searxng_base_url:
            providers.append("searxng")
        providers.append("jina")  # Free tier always available
        return providers


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
