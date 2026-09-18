"""RelayAI Python Brain Service.

A self-contained FastAPI companion service that mirrors the RelAI engine's
server-side brain: multi-provider LLM fallback, persistent memory (file +
Qdrant), RAG knowledge retrieval, and the empathetic companion turn loop.

Run it with `uvicorn companion.main:app --reload --port 8088` or `python -m companion`.
"""
__version__ = "1.0.0"
from . import brain_agent  # noqa: F401 - mounts upgraded agent routes
from .ingest import routes as ingest_routes  # noqa: F401 - mounts ingest routes
from . import device_routes  # noqa: F401 - mounts authenticated Device Host routes
