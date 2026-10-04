"""Desktop provider preferences, Windows-protected keys and bounded chat calls."""
from __future__ import annotations

import asyncio
import base64
import ctypes
import ipaddress
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import httpx

from companion.config import BrainSettings
from companion.providers import ProviderChain, ProviderResult

PROVIDERS = ("auto", "ollama", "openai", "gemini", "anthropic")
CLOUD_PROVIDERS = ("openai", "gemini", "anthropic")
CONFIG_NAME = "providers.json"
_CONFIG_FIELDS = ("provider", "model", "vision_provider", "vision_model", "ollama_base_url")


def _crypt(data: bytes, *, decrypt: bool = False) -> bytes:
    """DPAPI binds saved credentials to this Windows account; never store plaintext."""
    if sys.platform != "win32":
        raise RuntimeError("Saving API keys is supported on Windows. Use environment variables on this platform.")
    from ctypes import wintypes

    class Blob(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_ubyte))]

    buffer = ctypes.create_string_buffer(data)
    source = Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    target = Blob()
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    if decrypt:
        function = crypt32.CryptUnprotectData
        function.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.c_void_p,
                             ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob)]
        arguments = (ctypes.byref(source), None, None, None, None, 1, ctypes.byref(target))
    else:
        function = crypt32.CryptProtectData
        function.argtypes = [ctypes.POINTER(Blob), wintypes.LPCWSTR, ctypes.c_void_p,
                             ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob)]
        arguments = (ctypes.byref(source), "Sweep provider credential", None, None, None, 1, ctypes.byref(target))
    function.restype = wintypes.BOOL
    if not function(*arguments):
        raise RuntimeError("Windows could not unlock the saved API key. Re-enter it for this Windows account.")
    try:
        return ctypes.string_at(target.data, target.size)
    finally:
        kernel32.LocalFree(target.data)


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
    preferences = {"provider": "auto", "model": "", "vision_provider": "auto", "vision_model": "",
                   "ollama_base_url": settings.ollama_base_url}
    preferences.update({name: document[name] for name in _CONFIG_FIELDS if name in document})
    for name in ("provider", "vision_provider"):
        if preferences[name] not in PROVIDERS:
            raise ValueError("Choose a supported provider: Auto, Ollama, OpenAI, Gemini or Anthropic.")
    for name in ("model", "vision_model"):
        value = preferences[name]
        if not isinstance(value, str) or (value and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./:@+\-]{0,255}", value)):
            raise ValueError("Model names must be at most 256 characters and contain no spaces or control characters.")
    url = preferences["ollama_base_url"]
    if not isinstance(url, str) or len(url) > 2048 or any(ord(char) < 33 for char in url):
        raise ValueError("Enter a valid Ollama HTTP or HTTPS server address.")
    try:
        parsed = urlparse(url)
        valid = (parsed.scheme in {"http", "https"} and parsed.hostname and not parsed.username
                 and not parsed.password and not parsed.query and not parsed.fragment
                 and (parsed.port is None or 1 <= parsed.port <= 65535))
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("Enter an Ollama HTTP or HTTPS address without credentials, query parameters or fragments.")
    preferences["ollama_base_url"] = url.rstrip("/")
    return preferences


def _keys(document: dict, settings: BrainSettings) -> dict[str, str]:
    keys = {name: getattr(settings, f"{name}_api_key") for name in CLOUD_PROVIDERS}
    protected = document.get("protected_keys", {})
    if not isinstance(protected, dict):
        raise RuntimeError("Saved API keys could not be read. Re-enter the keys in provider settings.")
    for name in CLOUD_PROVIDERS:
        if name not in protected:
            continue
        try:
            encoded = base64.b64decode(protected[name], validate=True)
            keys[name] = _crypt(encoded, decrypt=True).decode("utf-8")
        except (ValueError, TypeError, RuntimeError, UnicodeError):
            raise RuntimeError(f"The saved {name} key is unavailable. Re-enter it for this Windows account.") from None
    return keys


def load_config(root: Path) -> dict:
    """Return settings and credential-presence flags; never return API key values."""
    settings = _base_settings(root)
    document = _read_document(root)
    config = _preferences(document, settings)
    try:
        keys = _keys(document, settings)
        config["key_present"] = {name: bool(value) for name, value in keys.items()}
        config["key_error"] = ""
    except RuntimeError as exc:
        config["key_present"] = {name: bool(getattr(settings, f"{name}_api_key")) for name in CLOUD_PROVIDERS}
        config["key_error"] = str(exc)
    return config


def save_config(root: Path, config: dict, keys: dict[str, str | None] | None = None) -> dict:
    """Atomically save preferences and DPAPI blobs. Missing/None keys are retained."""
    root = Path(root)
    settings = _base_settings(root)
    document = _read_document(root)
    preferences = _preferences({**document, **{name: config[name] for name in _CONFIG_FIELDS if name in config}}, settings)
    protected = dict(document.get("protected_keys", {}))
    for name, secret in (keys or {}).items():
        if name not in CLOUD_PROVIDERS:
            raise ValueError("API keys can be saved only for OpenAI, Gemini and Anthropic.")
        if secret is None:
            continue
        if not isinstance(secret, str) or len(secret) > 16_000 or any(ord(char) < 32 for char in secret):
            raise ValueError("Enter an API key without control characters.")
        secret = secret.strip()
        if secret:
            protected[name] = base64.b64encode(_crypt(secret.encode("utf-8"))).decode("ascii")
        else:
            protected.pop(name, None)
    output = {"schema_version": 1, **preferences, "protected_keys": protected}
    root.mkdir(parents=True, exist_ok=True)
    path = root / CONFIG_NAME
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return load_config(root)


def resolve_settings(root: Path) -> BrainSettings:
    """Merge environment settings with desktop choices and protected credentials."""
    settings = _base_settings(root)
    document = _read_document(root)
    config = _preferences(document, settings)
    for name, secret in _keys(document, settings).items():
        setattr(settings, f"{name}_api_key", secret)
    settings.ollama_base_url = config["ollama_base_url"]
    settings.request_timeout_seconds = (180.0 if config["provider"] == "ollama"
                                        else min(45.0, max(1.0, settings.request_timeout_seconds)))
    provider = config["provider"]
    if provider != "auto":
        settings.provider_order = [provider]
    else:
        settings.provider_order = [name for name in settings.provider_order if name in PROVIDERS and name != "auto"]
    if config["model"] and provider != "auto":
        field = {"gemini": "relai_model", "openai": "openai_model", "anthropic": "anthropic_model", "ollama": "ollama_model"}[provider]
        setattr(settings, field, config["model"])
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
                      and "embed" not in row["name"].lower()]
        candidates.sort(key=lambda row: (row.get("size") if isinstance(row.get("size"), (int, float)) else float("inf"), row["name"]))
        names = list(dict.fromkeys(row["name"] for row in candidates))
        return {"checked": True, "reachable": True, "models": names,
                "message": "Installed Ollama models are available." if names else "Ollama is running but has no installed chat model."}
    except (httpx.HTTPError, TimeoutError, ValueError, AttributeError):
        return {"checked": True, "reachable": False, "models": [],
                "message": "Ollama is not responding. Start Ollama and select an installed model, or configure a cloud provider."}


async def readiness(root: Path, *, check_local: bool = False) -> dict:
    """Summarize configuration; optional health check contacts loopback Ollama only."""
    config = load_config(root)
    available = [name for name, present in config["key_present"].items() if present]
    local = {"checked": False, "reachable": None, "models": [], "message": "Local Ollama has not been checked."}
    if check_local:
        local = await _ollama_models(config["ollama_base_url"])
    return {"provider": config["provider"], "configured_providers": available,
            "key_present": config["key_present"], "key_error": config["key_error"], "ollama": local,
            "message": "Cloud credentials are configured but have not been validated." if available else local["message"]}


async def start_local_server(root: Path) -> dict:
    """Start an already-installed Ollama on loopback only, without model downloads."""
    config = load_config(root)
    url = config["ollama_base_url"]
    parsed = urlparse(url)
    if not _is_loopback(url) or parsed.path not in {"", "/"} or parsed.scheme != "http":
        raise RuntimeError("Sweep can start only a local HTTP Ollama server. Start remote or custom servers yourself.")
    current = await _ollama_models(url)
    if current["reachable"]:
        return {"message": "Ollama is already running.", "ollama": current}
    candidates = [Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Ollama/ollama.exe"]
    found = shutil.which("ollama")
    if found:
        candidates.append(Path(found))
    executable = next((path for path in candidates if path.is_absolute() and path.is_file()), None)
    if executable is None:
        raise RuntimeError("Ollama is not installed. Install it and a chat model, or configure a cloud provider in Settings.")
    environment = os.environ.copy()
    hostname = parsed.hostname if parsed.hostname != "localhost" else "127.0.0.1"
    environment["OLLAMA_HOST"] = f"{'[' + hostname + ']' if ':' in hostname else hostname}:{parsed.port or 11434}"
    subprocess.Popen([str(executable), "serve"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, env=environment,
                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), start_new_session=os.name != "nt")
    for _ in range(10):
        await asyncio.sleep(.5)
        current = await _ollama_models(url)
        if current["reachable"]:
            return {"message": "Local Ollama is running. No models were downloaded.", "ollama": current}
    raise RuntimeError("Ollama did not become ready. Open Ollama directly and check its local server settings.")


async def generate_chat(root: Path, *, system: str, messages: list[dict], json_mode: bool = False) -> ProviderResult:
    """Use only the selected provider, or the configured fallback chain in Auto mode."""
    settings = resolve_settings(root)
    config = load_config(root)
    explicit = config["provider"] != "auto"
    if explicit and config["provider"] in CLOUD_PROVIDERS and not getattr(settings, f"{config['provider']}_api_key"):
        raise RuntimeError(f"Add a {config['provider']} API key in provider settings before sending a message.")
    if "ollama" in settings.provider_order and _is_loopback(settings.ollama_base_url):
        local = await _ollama_models(settings.ollama_base_url)
        models = local["models"]
        matches = settings.ollama_model in models or f"{settings.ollama_model}:latest" in models
        if models and not matches:
            if explicit and config["model"]:
                raise RuntimeError("The selected Ollama model is not installed. Choose one of the installed models in provider settings.")
            settings.ollama_model = models[0]
        elif not models:
            if explicit:
                raise RuntimeError(local["message"])
            settings.provider_order = [name for name in settings.provider_order if name != "ollama"]
    settings.provider_order = [name for name in settings.provider_order
                               if name == "ollama" or bool(getattr(settings, f"{name}_api_key", ""))]
    if not settings.provider_order:
        raise RuntimeError("No conversation provider is ready. Start Ollama with an installed chat model, or add a cloud API key in provider settings.")
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
    timeout = 180 if "ollama" in settings.provider_order else settings.request_timeout_seconds
    try:
        async with asyncio.timeout(timeout):
            return await ProviderChain(settings).generate(
                system=system, messages=list(reversed(bounded)), json_mode=json_mode, max_tokens=600,
                preferred=config["provider"] if explicit else None,
            )
    except TimeoutError:
        raise RuntimeError("The conversation provider timed out. Try a smaller local model or check the selected provider's connection.") from None
    except Exception:
        raise RuntimeError("The selected conversation provider could not respond. Check its API key, model name and connection in provider settings.") from None
