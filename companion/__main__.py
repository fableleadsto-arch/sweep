"""Entrypoint: `python -m companion`.

Runs the FastAPI brain service with uvicorn on `COMPANION_PORT` (default 8088).
"""

from __future__ import annotations

import os

import uvicorn

from .config import get_settings


def main() -> None:
    settings = get_settings()
    host = os.environ.get("COMPANION_HOST", "127.0.0.1")
    port = int(os.environ.get("COMPANION_PORT", str(settings.companion_port)))
    uvicorn.run("companion.main:app", host=host, port=port, reload=False, log_level="info")


if __name__ == "__main__":
    main()
