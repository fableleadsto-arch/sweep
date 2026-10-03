from importlib import metadata

from fastapi import FastAPI

from sweep_core import __version__
from sweep_core.config import Settings, get_settings
from sweep_core.fastpath import NATIVE_AVAILABLE
from sweep_core.logging_setup import setup_logging


def _version() -> str:
    try:
        return metadata.version("sweep-core")
    except metadata.PackageNotFoundError:
        return __version__


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or get_settings()
    setup_logging(app_settings)

    app = FastAPI(
        title="Sweep",
        version=_version(),
        description="Visual and voice recognition service",
    )

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "env": app_settings.env,
            "native_available": NATIVE_AVAILABLE,
        }

    @app.get("/version")
    def version() -> dict:
        return {"name": "sweep", "version": _version()}

    @app.get("/capabilities")
    def caps() -> dict:
        from sweep_core.integrations import capabilities

        return capabilities()

    return app


app = create_app()


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "sweep_core.api:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
