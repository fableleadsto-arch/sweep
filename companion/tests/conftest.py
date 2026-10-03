"""Shared pytest fixtures for the brain service tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the `companion` package importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from companion.config import BrainSettings  # noqa: E402
from companion.memory import FileMemoryStore, MemoryService  # noqa: E402


@pytest.fixture(autouse=True)
def isolate_api_configuration(monkeypatch):
    from companion.config import get_settings
    from companion.main import app

    monkeypatch.setenv("BRAIN_SERVICE_TOKEN", "test-only")
    monkeypatch.setenv("INGEST_SCHEDULER_ENABLED", "false")
    get_settings.cache_clear()
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()
    get_settings.cache_clear()


@pytest.fixture()
def settings(tmp_path: Path) -> BrainSettings:
    """Minimal settings — no API keys, isolated memory file, no `.env`."""
    return BrainSettings(
        _env_file=None,
        brain_service_token="test-only",
        supabase_url="",
        supabase_service_key="",
        gemini_api_key="",
        openai_api_key="",
        anthropic_api_key="",
        qdrant_api_url="",
        memory_file=str(tmp_path / ".relayhub" / "relai-memory.json"),
    )


@pytest.fixture()
def file_store(settings: BrainSettings) -> FileMemoryStore:
    return FileMemoryStore(settings)


@pytest.fixture()
def memory_service(settings: BrainSettings) -> MemoryService:
    return MemoryService(settings)
