"""Central configuration for the RelayAI brain service.

Every tunable lives here, driven by environment variables (mirrors
`.env.example`). Values are validated and typed at boot so a misconfigured
deployment fails loudly instead of silently degrading.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class BrainSettings(BaseSettings):
    """Typed, validated configuration for the Python brain service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Storage ──────────────────────────────────────────────────────
    # The TypeScript stack uses SUPABASE_SERVICE_ROLE_KEY; the legacy
    # companion used SUPABASE_SERVICE_KEY. Accept both so the service works
    # wherever it is deployed.
    supabase_url: str = ""
    supabase_service_key: str = ""
    supabase_service_role_key: str = ""

    # ── AI providers ─────────────────────────────────────────────────
    gemini_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    ollama_embed_model: str = "nomic-embed-text"
    tgi_base_url: str = ""
    langflow_base_url: str = ""
    langflow_api_key: str = ""

    # Model names — env overrides respected (RELAI_MODEL, OPENAI_MODEL, ...)
    # RELAI_MODEL is the canonical override used across the TS stack; the
    # gemini provider prefers it over COMPANION_MODEL when both are set.
    relai_model: str = ""
    companion_model: str = "gemini-2.0-flash"
    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-sonnet-4-6"
    ollama_model: str = "llama3.1:8b"

    # Preferred fallback order (names of provider classes).
    provider_order: list[str] = ["gemini", "openai", "ollama", "anthropic"]

    # ── Embeddings ───────────────────────────────────────────────────
    # gemini-embedding-001 is the current Gemini embedding model (the old
    # text-embedding-004 was retired). Dimensionality is pinned to 768 to stay
    # byte-compatible with existing Qdrant collections and the TS stack.
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 768
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = 90

    # ── Memory ───────────────────────────────────────────────────────
    mem0_api_key: str = ""
    mem0_user_id: str = ""
    qdrant_api_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "relai_memories"
    qdrant_vector_size: int = 768
    memory_file: str = ".relayhub/relai-memory.json"
    memory_cache_ttl_ms: int = 2_000
    memory_recency_half_life_days: float = 30.0

    # ── Automations ──────────────────────────────────────────────────
    n8n_webhook_base: str = ""

    # ── Relay context / corpus feeds ─────────────────────────────────
    # Kill-switches for the live workspace context (profile, overview,
    # snapshot, graph facts) and the shared corpus RAG. Both default on and
    # degrade to empty results whenever Supabase is unavailable.
    enable_relay_context: bool = True
    enable_corpus_search: bool = True

    # ── Capability engine ─────────────────────────────────────────────
    # When on, analysis/data/image requests are routed through the compute
    # toolbox (NumPy/Pandas/SymPy/OpenCV/sklearn/...) and the result is fed
    # into the LLM turn. Never loads a heavy framework for unrelated turns.
    enable_compute: bool = True
    # Static filtering is not an OS sandbox. Enable only for trusted scripts.
    allow_code_execution: bool = False
    cors_origins: str = ""

    # Backend enable/disable state for the compute layer (companion/compute/).
    # Persisted as JSON; defaults to enabling every backend and gating on the
    # actual hardware/frameworks present.
    compute_config_file: str = ".relayhub/compute-config.json"

    # When on, the `wheel-install` capability may `pip install` a wheel that
    # is stored in the vendored bundle registry (companion/vendor/archives/).
    # This mutates the Python environment, so it defaults to OFF — enable with
    # COMPANION_ALLOW_WHEEL_INSTALL=1 only on hosts where self-provisioning
    # is wanted. The registry is locked: arbitrary paths/URLs are never
    # accepted.
    allow_wheel_install: bool = False

    # ── Service ──────────────────────────────────────────────────────
    companion_port: int = 8088
    request_timeout_seconds: float = 90.0

    # ── Knowledge ingestion ───────────────────────────────────────────
    # Continuous ingestion of free/public sources into the global knowledge
    # store. `ingest_store` picks the persistence backend: "auto" uses
    # Supabase/PostgreSQL when the service key is configured, otherwise the
    # local JSON fallback under `.relayhub/ingest/`.
    enable_knowledge_ingestion: bool = True
    ingest_store: str = "auto"
    ingest_data_dir: str = ".relayhub/ingest"
    ingest_scheduler_enabled: bool = False
    ingest_scheduler_tick_seconds: int = 60
    ingest_max_documents_per_source: int = 20
    # Relevance below this is rejected before any embedding/extraction work.
    ingest_min_relevance: float = 0.2
    ingest_embed_when_available: bool = True
    ingest_http_timeout_seconds: float = 20.0
    ingest_user_agent: str = "RelayAI-KnowledgeBot/1.0 (+https://github.com/relai)"

    # Optional tokens (rate-limit upgrades only — never required).
    github_token: str = ""
    openalex_api_key: str = ""

    # ── Auth ─────────────────────────────────────────────────────────
    # Required bearer token. Every /api/* request must send
    # `Authorization: Bearer <token>` or it is rejected with 401. Health and
    # the OpenAPI docs stay open so uptime probes work.
    brain_service_token: str = ""

    # ── Derived helpers ──────────────────────────────────────────────

    @field_validator("provider_order", mode="before")
    @classmethod
    def _split_provider_order(cls, value: object) -> object:
        if isinstance(value, str):
            return [p.strip() for p in value.split(",") if p.strip()]
        return value

    @property
    def supabase_key(self) -> str:
        """Resolve the service key from either supported env var name."""
        return self.supabase_service_key or self.supabase_service_role_key

    @property
    def gemini_model(self) -> str:
        """Active Gemini model — RELAI_MODEL wins, then COMPANION_MODEL."""
        return self.relai_model or self.companion_model

    @property
    def has_ai_provider(self) -> bool:
        return bool(
            self.gemini_api_key
            or self.openai_api_key
            or self.anthropic_api_key
            or self.ollama_base_url
        )

    @property
    def qdrant_configured(self) -> bool:
        return bool(self.qdrant_api_url)

    @property
    def mem0_configured(self) -> bool:
        return bool(self.mem0_api_key)


@lru_cache(maxsize=1)
def get_settings() -> BrainSettings:
    """Return the process-wide settings singleton."""
    return BrainSettings()
