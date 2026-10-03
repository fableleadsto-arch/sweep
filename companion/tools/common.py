"""Shared helpers for the capability tool runners.

Keeps the per-domain tool modules (`numeric`, `data`, `ml`, ...) free of
plumbing: payload normalization, lazy imports with helpful errors, and a
structured "unavailable" outcome for optional frameworks that are not
installed in this environment.
"""

from __future__ import annotations

import importlib
import importlib.util
import re
from typing import Any, Optional

from .vendor_loader import add_vendored_paths, is_vendored, vendored_path

# Make vendored frameworks importable (they live at companion/vendor/). This
# appends to sys.path so pip-installed packages always win.
add_vendored_paths()

# The set of capabilities that require a heavy/optional framework. When one of
# these modules is missing the engine reports a clear install hint instead of
# a confusing import traceback.
OPTIONAL_FRAMEWORKS = {
    "torch": "PyTorch",
    "tensorflow": "TensorFlow",
    "keras": "Keras",
    "jax": "JAX",
    "transformers": "Hugging Face Transformers",
    "diffusers": "Hugging Face Diffusers",
    "accelerate": "Hugging Face Accelerate",
    "timm": "pytorch-image-models (timm)",
    "litellm": "LiteLLM",
    "llama_index": "LlamaIndex",
    "langchain": "LangChain",
    "crewai": "crewAI",
    "autogen": "Microsoft autogen",
    "qdrant_client": "Qdrant client",
    "pymilvus": "Milvus (pymilvus)",
    "vllm": "vLLM",
    "llama_cpp": "llama-cpp-python",
    "onnxruntime": "ONNX Runtime",
    "tensorrt": "NVIDIA TensorRT",
    "tensorrt_llm": "NVIDIA TensorRT-LLM",
    "triton": "Triton (GPU kernels)",
    "deepspeed": "DeepSpeed",
    "megatron": "Megatron-LM",
    "axolotl": "axolotl",
    "bitsandbytes": "bitsandbytes",
    "unsloth": "unsloth",
    "cv2": "OpenCV",
    "spacy": "spaCy",
    "nltk": "NLTK",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}


def module_available(name: str) -> bool:
    """Cheap availability probe — never imports the module.

    A framework is available when it is pip-installed **or** its source is
    vendored in-repo (companion/vendor/). Vendored *archives* of compiled
    giants (torch, tensorflow, ...) do not count — they need a native build.
    """
    if importlib.util.find_spec(name) is not None:
        return True
    # Installed wins; only fall back to vendored source when it is importable
    # (the vendored root is already on sys.path via add_vendored_paths).
    return is_vendored(name) and vendored_path(name) is not None


def has_cuda() -> bool:
    """True when a CUDA-capable torch build sees a GPU. Never raises."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - torch missing/import error
        return False


def is_safe_http_url(raw: str) -> bool:
    """Validate URL syntax/address literals; fetchers must also validate DNS/redirects."""
    from app.core.guard import validate_safe_url

    return isinstance(raw, str) and validate_safe_url(raw)[0]


def load(name: str):
    """Lazily import a module, raising a friendly error when it is missing.

    Resolves pip-installed first; falls back to the vendored source tree when
    the framework is vendored in-repo but not installed.

    Vendored pure-Python trees still need their real pip dependencies (e.g.
    vendored transformers needs torch/tokenizers). If an import fails for a
    missing dependency, that is surfaced as a CapabilityUnavailable naming the
    actual missing module — not a raw traceback — so the catalog's availability
    stays honest.
    """
    if not module_available(name):
        pretty = OPTIONAL_FRAMEWORKS.get(name, name)
        raise CapabilityUnavailable(
            f"{pretty} is not installed in this environment. "
            f"Install it with:  pip install -r requirements.companion-ai.txt  "
            f"(or the per-backend profile, e.g. requirements.companion-pytorch.txt, "
            f"for just this framework)."
        )
    try:
        return importlib.import_module(name)
    except (ImportError, ModuleNotFoundError) as exc:
        missing = getattr(exc, "name", None) or str(exc).split("'")[1] if "'" in str(exc) else name
        pretty = OPTIONAL_FRAMEWORKS.get(str(missing), str(missing))
        raise CapabilityUnavailable(
            f"{OPTIONAL_FRAMEWORKS.get(name, name)} is present (installed or vendored) but "
            f"cannot be imported because its dependency '{pretty}' is missing. "
            f"Install the dependency with:  pip install {missing}"
        ) from exc


class CapabilityUnavailable(Exception):
    """Raised when a capability's required framework is not installed."""


def unavailable_result(capability: str, libraries: list[str]) -> dict[str, Any]:
    """Structured 'framework not installed' outcome for a capability."""
    missing = [lib for lib in libraries if not module_available(lib)]
    return {
        "result": {
            "capability": capability,
            "ready": False,
            "missing_libraries": missing,
        },
        "summary": (
            f"The {capability} capability needs {', '.join(missing)} which are "
            f"not installed. Run `pip install -r requirements.companion-ai.txt` "
            f"to enable it."
        ),
        "libraries_used": [],
    }


# ── payload normalization ────────────────────────────────────────────────

def as_numbers(data: Any) -> list[float]:
    """Coerce data into a flat list of numbers."""
    if data is None:
        return []
    if isinstance(data, (int, float)):
        return [float(data)]
    if isinstance(data, str):
        values = [v for v in re.split(r"[,\s;\n]+", data.strip()) if v]
        numbers: list[float] = []
        for v in values:
            try:
                numbers.append(float(v))
            except ValueError:
                continue
        return numbers
    if isinstance(data, dict):
        values = [
            v
            for k, v in data.items()
            if isinstance(v, (int, float)) or (isinstance(v, str) and v.replace(".", "", 1).isdigit())
        ]
        return [float(v) for v in values]
    if isinstance(data, (list, tuple)):
        flat: list[float] = []
        for item in data:
            if isinstance(item, (list, tuple)):
                for sub in item:
                    if isinstance(sub, (int, float)):
                        flat.append(float(sub))
            elif isinstance(item, (int, float)):
                flat.append(float(item))
            elif isinstance(item, str) and item.replace(".", "", 1).replace("-", "", 1).isdigit():
                flat.append(float(item))
        return flat
    return []


def as_matrix(data: Any) -> Optional[list[list[float]]]:
    """Coerce data into a numeric 2-D matrix, or None when not rectangular."""
    if not isinstance(data, list) or not data:
        return None
    rows: list[list[float]] = []
    for row in data:
        if isinstance(row, dict):
            vals = [v for v in row.values() if isinstance(v, (int, float))]
            rows.append(vals)
        elif isinstance(row, (list, tuple)):
            vals = [float(v) for v in row if isinstance(v, (int, float))]
            rows.append(vals)
        elif isinstance(row, (int, float)):
            rows.append([float(row)])
        else:
            return None
    width = len(rows[0])
    if width == 0 or any(len(r) != width for r in rows):
        return None
    return rows


def as_rows(data: Any) -> tuple[Optional[list[str]], list[list[Any]]]:
    """Coerce data into (columns, rows) for tabular analysis.

    Accepts a CSV string, a list of dicts, a list of lists, or a dict shaped
    like ``{"columns": [...], "rows": [...]}``.
    """
    if data is None:
        return None, []
    if isinstance(data, str):
        csv_text = data.strip()
        if "\n" not in csv_text and "," not in csv_text:
            return None, []
        import io

        try:
            import csv

            reader = csv.reader(io.StringIO(csv_text))
            table = [row for row in reader if any(cell.strip() for cell in row)]
        except Exception:  # noqa: BLE001 - malformed CSV degrades to empty
            return None, []
        if not table:
            return None, []
        header = [c.strip() for c in table[0]]
        body = table[1:]
        # If the header row is clearly data (e.g. all numeric), treat it as
        # a headerless table.
        try:
            [float(c) for c in header]
            body = table
            header = [f"col{i}" for i in range(len(header))]
        except ValueError:
            pass
        return header, [list(row) for row in body]
    if isinstance(data, dict):
        if isinstance(data.get("rows"), list):
            columns = data.get("columns")
            columns = list(columns) if isinstance(columns, list) else None
            return columns, [list(r) for r in data["rows"]]
        return None, []
    if isinstance(data, list):
        if not data:
            return None, []
        first = data[0]
        if isinstance(first, dict):
            columns = list(first.keys())
            return columns, [list(row.values()) for row in data if isinstance(row, dict)]
        if isinstance(first, (list, tuple)):
            return None, [list(row) for row in data]
    return None, []


def as_text(data: Any, params: dict[str, Any]) -> str:
    """Coerce data into a plain text string for NLP."""
    if isinstance(params.get("text"), str):
        return params["text"]
    if isinstance(data, str):
        return data
    if isinstance(data, list):
        parts = []
        for item in data:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(" ".join(str(v) for v in item.values() if v is not None))
        return "\n".join(parts)
    if isinstance(data, dict):
        return " ".join(str(v) for v in data.values() if v is not None)
    return str(data or "")
