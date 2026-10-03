"""RelayAI Python Brain Service.

A self-contained FastAPI companion service that mirrors the RelAI engine's
server-side brain: multi-provider LLM fallback, persistent memory (file +
Qdrant), RAG knowledge retrieval, and the empathetic companion turn loop.

Run it with:

    uvicorn companion.main:app --reload --port 8088

or the convenience entrypoint:

    python -m companion
"""

__version__ = "1.0.0"

# Importing utility modules must not boot the HTTP service. Routes are mounted
# explicitly in companion.main.
