"""Local desktop intelligence: bounded inference with no cloud fallback."""
from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import httpx

from companion.config import BrainSettings
from companion.providers import ProviderChain, ProviderResult

PROVIDERS = ("auto", "ollama", "openai", "gemini", "anthropic")  # legacy preference migration
CLOUD_PROVIDERS = ("openai", "gemini", "anthropic")
CONFIG_NAME = "providers.json"
_CONFIG_FIELDS = ("provider", "model", "vision_provider", "vision_model", "ollama_base_url")


def _base_settings(root: Path) -> BrainSettings:
    # Desktop startup already imports the development .env into its environment.
    # Installed apps read only their own data directory and process environment.
    return BrainSettings(_env_file=Path(root) / ".env")


def _read_document(root: Path) -> dict:
    path = Path(root) / CONFIG_NAME
    if not path.exists():
        return {}
    try:
        if path.stat().st_size > 100_000:
            raise ValueError("oversized settings")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            raise ValueError("unsupported settings")
        return data
    except (OSError, ValueError):
        raise RuntimeError("Saved provider settings could not be read. Restore or remove providers.json and save the settings again.") from None


def _preferences(document: dict, settings: BrainSettings) -> dict:
    preferences = {"provider": "ollama", "model": "", "vision_provider": "ollama", "vision_model": "",
                   "ollama_base_url": settings.ollama_base_url}
    preferences.update({name: document[name] for name in _CONFIG_FIELDS if name in document})
    for name in ("provider", "vision_provider"):
        if preferences[name] not in PROVIDERS:
            raise ValueError("Invalid local intelligence settings.")
    # Older preferences can name cloud services; migrate them without reading
    # credentials or forwarding a cloud model name to the local engine.
    for provider, model in (("provider", "model"), ("vision_provider", "vision_model")):
        if preferences[provider] in CLOUD_PROVIDERS:
            preferences[model] = ""
        preferences[provider] = "ollama"
    for name in ("model", "vision_model"):
        value = preferences[name]
        if not isinstance(value, str) or (value and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./:@+\-]{0,255}", value)):
            raise ValueError("Model names must be at most 256 characters and contain no spaces or control characters.")
        if re.search(r"(?:^|[:/_-])cloud(?:$|[:/_-])", value, re.I):
            raise ValueError("Sweep uses downloaded local models. Cloud models are not allowed.")
    url = preferences["ollama_base_url"]
    if not isinstance(url, str) or len(url) > 2048 or any(ord(char) < 33 for char in url):
        raise ValueError("Enter a valid local HTTP server address.")
    try:
        parsed = urlparse(url)
        valid = (parsed.scheme == "http" and parsed.hostname and _is_loopback(url) and not parsed.username
                 and not parsed.password and not parsed.query and not parsed.fragment
                 and parsed.path in {"", "/"} and "\\" not in url
                 and (parsed.port is None or 1 <= parsed.port <= 65535))
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("Local processing requires an HTTP loopback address on this computer, without credentials or a path.")
    host = "127.0.0.1" if parsed.hostname.lower() == "localhost" else parsed.hostname
    preferences["ollama_base_url"] = f"http://{'[' + host + ']' if ':' in host else host}:{parsed.port or 11434}"
    return preferences


def load_config(root: Path) -> dict:
    """Return local-only preferences; older protected credentials stay unused."""
    settings = _base_settings(root)
    document = _read_document(root)
    config = _preferences(document, settings)
    config["local_only"] = True
    config["key_present"] = {name: False for name in CLOUD_PROVIDERS}
    config["key_error"] = ""
    return config


def save_config(root: Path, config: dict, keys: dict[str, str | None] | None = None) -> dict:
    """Save local preferences without touching older encrypted credentials."""
    root = Path(root)
    settings = _base_settings(root)
    document = _read_document(root)
    if any(config.get(name) in CLOUD_PROVIDERS for name in ("provider", "vision_provider")) or any((keys or {}).values()):
        raise ValueError("Sweep desktop processes conversations and files locally. Cloud providers and API keys are disabled.")
    preferences = _preferences({**document, **{name: config[name] for name in _CONFIG_FIELDS if name in config}}, settings)
    protected = dict(document.get("protected_keys", {}))
    output = {"schema_version": 1, **preferences, "protected_keys": protected}
    root.mkdir(parents=True, exist_ok=True)
    path = root / CONFIG_NAME
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return load_config(root)


def resolve_settings(root: Path) -> BrainSettings:
    """Ignore environment cloud credentials and allow only local inference."""
    settings = _base_settings(root)
    document = _read_document(root)
    config = _preferences(document, settings)
    for name in CLOUD_PROVIDERS:
        setattr(settings, f"{name}_api_key", "")
    settings.ollama_base_url = config["ollama_base_url"]
    settings.request_timeout_seconds = 180.0
    settings.provider_order = ["ollama"]
    if config["model"]:
        settings.ollama_model = config["model"]
    return settings


def _is_loopback(url: str) -> bool:
    host = urlparse(url).hostname or ""
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


async def _ollama_models(base_url: str) -> dict:
    """Inspect only an explicitly local server, without downloading any models."""
    if not _is_loopback(base_url):
        return {"checked": False, "reachable": None, "models": [], "message": "Remote Ollama discovery is not performed automatically."}
    try:
        async with asyncio.timeout(3), httpx.AsyncClient(timeout=2, trust_env=False, follow_redirects=False) as client:
            async with client.stream("GET", f"{base_url.rstrip('/')}/api/tags") as response:
                response.raise_for_status()
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > 256_000:
                        raise ValueError("oversized model list")
        data = json.loads(body)
        rows = data.get("models", [])
        if not isinstance(rows, list):
            raise ValueError("invalid model list")
        candidates = [row for row in rows if isinstance(row, dict) and isinstance(row.get("name"), str)
                      and "embed" not in row["name"].lower() and "cloud" not in row["name"].lower()
                      and not row.get("remote_host") and not row.get("remote_model")]
        candidates.sort(key=lambda row: (row.get("size") if isinstance(row.get("size"), (int, float)) else float("inf"), row["name"]))
        names = list(dict.fromkeys(row["name"] for row in candidates))
        return {"checked": True, "reachable": True, "models": names,
                "message": "Local intelligence is ready." if names else "No downloaded local model is available. Open advanced settings to select one."}
    except (httpx.HTTPError, TimeoutError, ValueError, AttributeError):
        return {"checked": True, "reachable": False, "models": [],
                "message": "The local intelligence engine is not running. Start local processing in Settings."}


async def require_local_model(base_url: str, model: str, *, vision: bool = False) -> dict:
    """Reject cloud aliases before any conversation or image data is sent."""
    try:
        parsed = urlparse(base_url)
        local = (_is_loopback(base_url) and parsed.scheme == "http" and not parsed.username
                 and not parsed.password and parsed.path in {"", "/"} and not parsed.query
                 and not parsed.fragment and "\\" not in base_url
                 and not any(ord(char) < 33 for char in base_url)
                 and (parsed.port is None or 1 <= parsed.port <= 65535))
    except (TypeError, ValueError):
        local = False
    if not local or not isinstance(model, str) or not model or "cloud" in model.lower():
        raise RuntimeError("This task requires a downloaded model running on this computer.")
    try:
        async with asyncio.timeout(5), httpx.AsyncClient(timeout=4, trust_env=False, follow_redirects=False) as client:
            async with client.stream("POST", base_url.rstrip("/") + "/api/show", json={"model": model}) as response:
                response.raise_for_status()
                raw = bytearray()
                async for chunk in response.aiter_bytes():
                    raw.extend(chunk)
                    if len(raw) > 1_000_000:
                        raise ValueError("oversized model metadata")
        info = json.loads(raw)
        if (not isinstance(info, dict) or info.get("remote_host") or info.get("remote_model")
                or not isinstance(info.get("model_info"), dict) or not info["model_info"]):
            raise ValueError("remote or unverified model")
        capabilities = info.get("capabilities")
        if vision and (not isinstance(capabilities, list) or not all(isinstance(item, str) for item in capabilities)
                       or "vision" not in capabilities):
            raise RuntimeError("The selected local model cannot read images. Choose an image-capable model in advanced settings.")
        return info
    except (httpx.HTTPError, TimeoutError, ValueError):
        raise RuntimeError("Sweep could not verify a downloaded local model. Check local processing in Settings; no content was sent for analysis.") from None


async def readiness(root: Path, *, check_local: bool = False) -> dict:
    """Summarize configuration; optional health check contacts loopback Ollama only."""
    config = load_config(root)
    local = {"checked": False, "reachable": None, "models": [], "message": "Local Ollama has not been checked."}
    if check_local:
        local = await _ollama_models(config["ollama_base_url"])
    return {"provider": config["provider"], "configured_providers": ["local"], "local_only": True,
            "key_present": config["key_present"], "key_error": config["key_error"], "ollama": local,
            "message": local["message"]}


async def start_local_server(root: Path) -> dict:
    """Start an already-installed Ollama on loopback only, without model downloads."""
    config = load_config(root)
    url = config["ollama_base_url"]
    parsed = urlparse(url)
    if not _is_loopback(url) or parsed.path not in {"", "/"} or parsed.scheme != "http":
        raise RuntimeError("Sweep can start only a local HTTP Ollama server. Start remote or custom servers yourself.")
    current = await _ollama_models(url)
    if current["reachable"]:
        return {"message": "Local intelligence is already running.", "ollama": current}
    candidates = [Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Ollama/ollama.exe"]
    found = shutil.which("ollama")
    if found:
        candidates.append(Path(found))
    executable = next((path for path in candidates if path.is_absolute() and path.is_file()), None)
    if executable is None:
        raise RuntimeError("The local intelligence engine is not installed. Complete local setup before using conversation or image analysis.")
    environment = os.environ.copy()
    hostname = parsed.hostname if parsed.hostname != "localhost" else "127.0.0.1"
    environment["OLLAMA_HOST"] = f"{'[' + hostname + ']' if ':' in hostname else hostname}:{parsed.port or 11434}"
    environment["OLLAMA_NO_CLOUD"] = "1"
    environment["OLLAMA_KEEP_ALIVE"] = "1m"
    subprocess.Popen([str(executable), "serve"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, env=environment,
                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), start_new_session=os.name != "nt")
    for _ in range(10):
        await asyncio.sleep(.5)
        current = await _ollama_models(url)
        if current["reachable"]:
            return {"message": "Local intelligence is ready. Processing stays on this computer.", "ollama": current}
    raise RuntimeError("Local intelligence did not become ready. Check the local engine in advanced settings.")


async def generate_chat(root: Path, *, system: str, messages: list[dict], json_mode: bool = False) -> ProviderResult:
    """Generate locally; never fall back to a remote service."""
    settings = resolve_settings(root)
    config = load_config(root)
    local = await _ollama_models(settings.ollama_base_url)
    if local.get("reachable") is False:
        local = (await start_local_server(root))["ollama"]
    models = local["models"]
    matches = settings.ollama_model in models or f"{settings.ollama_model}:latest" in models
    if models and not matches:
        if config["model"]:
            raise RuntimeError("The selected local model is not installed. Choose a downloaded model in advanced settings.")
        choices = [name for name in models if name != config["vision_model"]] or models
        settings.ollama_model = choices[0]
    elif not models:
        raise RuntimeError(local["message"])
    await require_local_model(settings.ollama_base_url, settings.ollama_model)
    # Keep recent context useful on CPU models, instead of spending the whole
    # request processing complete search snippets or file previews.
    bounded = []
    remaining = 10_000
    for message in reversed(messages):
        limit = 6000 if message.get("role") == "user" else 2000
        content = str(message.get("content", ""))[:min(limit, remaining)]
        if not content:
            continue
        bounded.append({"role": message["role"], "content": content})
        remaining -= len(content)
        if remaining == 0:
            break
    timeout = 180
    try:
        async with asyncio.timeout(timeout):
            return await ProviderChain(settings).generate(
                system=system, messages=list(reversed(bounded)), json_mode=json_mode, max_tokens=600,
                preferred="ollama",
            )
    except TimeoutError:
        raise RuntimeError("Local processing took too long. Try a shorter request or a smaller installed model in advanced settings.") from None
    except Exception:
        raise RuntimeError("Local intelligence could not finish this request. Check its status in Settings and retry.") from None
