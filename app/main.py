"""Sweep — FastAPI backend for C++ terminal UI.

Pure API server. The C++ UI connects to this backend for data.
No web frontend, no templates, no static files.

    uvicorn app.main:app --reload --port 8787
    python -m app.main
"""

from __future__ import annotations

import logging
import sys
from importlib import metadata
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .api.routes import router
from .core.auth import authorize

logger = logging.getLogger("sweep")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    cpp_status = "available" if _check_cpp() else "not compiled (pure Python mode)"
    logger.info(f"[Sweep] C++ engine: {cpp_status}")
    logger.info(f"[Sweep] Search providers: {settings.search_providers_configured}")
    yield


def _check_cpp() -> bool:
    try:
        import sweep_engine
        return True
    except ImportError:
        return False


app = FastAPI(
    title="Sweep API",
    description="Python/C++ web intelligence backend",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
)

settings = get_settings()
cors_origins = settings.cors_origin_list
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

def require_token(authorization: str | None = Header(default=None)) -> None:
    authorize(authorization, get_settings().sweep_api_token)


app.include_router(router, dependencies=[Depends(require_token)])


@app.get("/health")
async def health():
    def installed_version(name: str) -> str:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return ""

    torch_version = installed_version("torch")
    tf_version = installed_version("tensorflow")
    return {
        "status": "ok",
        "service": "Sweep",
        "version": "2.0.0",
        "python": sys.version.split()[0],
        "cpp": _check_cpp(),
        "pytorch": bool(torch_version),
        "pytorch_version": torch_version,
        "tensorflow": bool(tf_version),
        "tensorflow_version": tf_version,
    }


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=settings.debug)
