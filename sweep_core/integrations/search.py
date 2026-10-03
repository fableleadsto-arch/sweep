"""Search engine integrations: Meilisearch, Typesense, Elasticsearch.

All three are client integrations against locally hosted servers.
``meili_server`` can launch a downloaded Meilisearch binary so the
stack is self-contained (see scripts/fetch_integration_assets.py).
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any

from sweep_core.integrations import _module_available

MEILI_BINARY = Path(os.environ.get("SWEEP_MEILI_BIN", "bin/meilisearch.exe"))
MEILI_URL = os.environ.get("SWEEP_MEILI_URL", "http://127.0.0.1:7700")
MEILI_KEY = os.environ.get("SWEEP_MEILI_KEY", "sweep-dev")


def availability() -> dict[str, Any]:
    return {
        "meilisearch": {
            "client": _module_available("meilisearch"),
            "binary": MEILI_BINARY.exists(),
        },
        "typesense": {"client": _module_available("typesense")},
        "elasticsearch": {"client": _module_available("elasticsearch")},
    }


def start_meili_server() -> dict[str, Any]:
    """Start the local Meilisearch binary if present; wait until healthy."""
    import urllib.error
    import urllib.request

    if not MEILI_BINARY.exists():
        raise FileNotFoundError(f"meilisearch binary not found at {MEILI_BINARY}")
    proc = subprocess.Popen(
        [str(MEILI_BINARY), "--http-addr", "127.0.0.1:7700", "--master-key", MEILI_KEY or "sweep-dev"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    for _ in range(50):
        time.sleep(0.2)
        try:
            with urllib.request.urlopen(f"{MEILI_URL}/health", timeout=1) as resp:
                if resp.status == 200:
                    return {"pid": proc.pid, "url": MEILI_URL, "healthy": True}
        except (urllib.error.URLError, OSError):
            continue
    proc.terminate()
    raise RuntimeError("meilisearch did not become healthy in time")


def meili_client() -> Any:
    import meilisearch

    return meilisearch.Client(MEILI_URL, MEILI_KEY or None)


def typesense_client(**overrides: Any) -> Any:
    import typesense

    options = {
        "nodes": [{"host": "127.0.0.1", "port": 8108, "protocol": "http"}],
        "api_key": os.environ.get("SWEEP_TYPESENSE_KEY", "sweep-dev"),
        "connection_timeout_seconds": 5,
    }
    options.update(overrides)
    return typesense.Client(options)


def elasticsearch_client(**overrides: Any) -> Any:
    from elasticsearch import Elasticsearch

    kwargs: dict[str, Any] = {"hosts": [os.environ.get("SWEEP_ES_URL", "http://127.0.0.1:9200")]}
    kwargs.update(overrides)
    return Elasticsearch(**kwargs)
