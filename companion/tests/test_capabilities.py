"""Tests for the capability engine — the computational-toolbox layer.

Covers the catalog, automatic selection, availability detection and real
execution through the installed frameworks. Heavy model downloads (Hugging
Face weights, llama-index embeddings) are intentionally NOT exercised here —
the engine only needs the frameworks present, which the core requirements
provide.
"""

from __future__ import annotations

import asyncio
import base64
import io

import pytest
from fastapi.testclient import TestClient

from companion.capabilities import (
    CAPABILITIES,
    CapabilityEngine,
    list_capabilities,
    resolve_capability,
)
from companion.config import BrainSettings, get_settings
from companion.main import app
from companion.schemas import ComputeRequest, ComputeResult

ALL_FRAMEWORKS = {
    "numpy": "NumPy",
    "scipy": "SciPy",
    "pandas": "Pandas",
    "sklearn": "Scikit-learn",
    "sympy": "SymPy",
    "matplotlib": "Matplotlib",
    "PIL": "Pillow",
    "networkx": "NetworkX",
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

SAMPLE_CSV = "name,age,city\nAlice,30,NY\nBob,25,SF\nCarol,35,NY"
SAMPLE_ROWS = [
    {"a": 1, "b": 2, "y": "yes"},
    {"a": 2, "b": 3, "y": "no"},
    {"a": 3, "b": 4, "y": "yes"},
    {"a": 4, "b": 5, "y": "no"},
    {"a": 5, "b": 6, "y": "yes"},
    {"a": 6, "b": 7, "y": "yes"},
    {"a": 7, "b": 8, "y": "no"},
    {"a": 8, "b": 9, "y": "yes"},
    {"a": 9, "b": 10, "y": "no"},
    {"a": 10, "b": 11, "y": "yes"},
    {"a": 11, "b": 12, "y": "yes"},
    {"a": 12, "b": 13, "y": "no"},
]


def _engine(settings: BrainSettings) -> CapabilityEngine:
    return CapabilityEngine(settings)


def _run(engine: CapabilityEngine, **kwargs) -> ComputeResult:
    return asyncio.run(engine.run(ComputeRequest(**kwargs)))


def _tiny_png() -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (200, 30, 30)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ── catalog & availability ──────────────────────────────────────────────

def test_catalog_covers_every_framework(settings: BrainSettings) -> None:
    catalog = list_capabilities()
    assert len(catalog) >= 35
    for cap in catalog:
        # Framework-free capabilities (vendor-source, wheel-install) declare no
        # libraries by design; everything else must declare at least one.
        if cap.id not in {"vendor-source", "wheel-install"}:
            assert cap.libraries, f"capability {cap.id} must declare libraries"

    # Every framework from the spec is registered under some capability.
    registered = {lib for cap in CAPABILITIES for lib in cap.libraries}
    for lib in ALL_FRAMEWORKS:
        assert lib in registered, f"{lib} is not registered in the catalog"


def test_availability_detection() -> None:
    engine = CapabilityEngine()
    catalog = {c.id: c for c in engine.catalog()}
    # Core frameworks are present in the test environment.
    assert catalog["math"].available
    assert catalog["data-analysis"].available
    # A made-up library is correctly reported as missing.
    from companion.tools.common import module_available

    assert not module_available("definitely-not-a-real-package")


def test_optional_frameworks_declared_but_not_required_for_startup() -> None:
    # Other suites may already have imported frameworks: check a clean process.
    import subprocess
    import sys
    code = (
        "import sys; from companion.capabilities import list_capabilities; list_capabilities(); "
        "heavy = {'torch', 'tensorflow', 'jax', 'transformers', 'llama_index', 'litellm', 'cv2', 'spacy'}; "
        "assert not heavy.intersection(sys.modules), heavy.intersection(sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


# ── automatic selection ─────────────────────────────────────────────────

def test_resolve_explicit_override() -> None:
    cap, _ = resolve_capability("anything at all", capability_id="symbolic")
    assert cap.id == "symbolic"


def test_resolve_unknown_capability_raises() -> None:
    with pytest.raises(ValueError):
        resolve_capability("x", capability_id="not-a-capability")


def test_resolve_by_keywords() -> None:
    assert resolve_capability("analyze this csv", data=SAMPLE_CSV)[0].id == "data-analysis"
    assert resolve_capability("solve x**2 - 4 = 0", data="x**2 - 4 = 0")[0].id == "symbolic"
    assert resolve_capability("train a model to predict")[0].id == "ml"
    assert resolve_capability("plot this data", data=[1, 2, 3])[0].id == "plot"
    assert resolve_capability("describe this image")[0].id == "vision"
    assert resolve_capability("simulate a random walk")[0].id == "simulation"
    assert resolve_capability("train an xgboost model")[0].id == "gradient-boost"
    assert resolve_capability("extract entities from this text")[0].id == "nlp"
    assert resolve_capability("shortest path in this graph")[0].id == "graph"
    assert resolve_capability("use litellm to call gemini")[0].id == "llm-gateway"
    assert resolve_capability("generate an image of a sunset")[0].id == "diffusion"
    assert resolve_capability("list ollama models")[0].id == "ollama"
    assert resolve_capability("run a gguf model with llama.cpp")[0].id == "llamacpp"
    assert resolve_capability("list onnx providers")[0].id == "onnx"
    assert resolve_capability("vllm health check")[0].id == "vllm"
    assert resolve_capability("build a langchain chain")[0].id == "langchain"
    assert resolve_capability("run a crewai crew")[0].id == "crewai"
    assert resolve_capability("search a qdrant collection")[0].id == "qdrant"
    assert resolve_capability("list milvus collections")[0].id == "milvus"
    assert resolve_capability("deepspeed health")[0].id == "deepspeed"
    assert resolve_capability("bitsandbytes quantization")[0].id == "bitsandbytes"
    assert resolve_capability("list timm models")[0].id == "timm"
    assert resolve_capability("autogen conversation")[0].id == "autogen"
    assert resolve_capability("accelerate device state")[0].id == "accelerate"


def test_resolve_by_data_shape() -> None:
    assert resolve_capability("look at this")[0].id == "math"
    assert resolve_capability("look at this", data=[1, 2, 3])[0].id == "math"
    assert resolve_capability("look at this", data=SAMPLE_ROWS)[0].id == "data-analysis"
    assert resolve_capability("look at this", data="a very long text string for nlp analysis")[0].id == "nlp"


# ── execution ───────────────────────────────────────────────────────────

def test_math_compute(settings: BrainSettings) -> None:
    result = _run(_engine(settings), task="calculate the mean and median", data=[1, 2, 3, 4, 5])
    assert result.ok
    assert result.capability == "math"
    assert result.result["stats"]["mean"] == 3.0
    assert result.result["stats"]["median"] == 3.0


def test_data_analysis_compute(settings: BrainSettings) -> None:
    result = _run(_engine(settings), task="analyze this csv", data=SAMPLE_CSV)
    assert result.ok
    assert result.capability == "data-analysis"
    assert result.result["rows"] == 3
    assert result.result["columns"] == 3
    assert "age" in result.result["numeric_summary"]


def test_symbolic_compute(settings: BrainSettings) -> None:
    result = _run(_engine(settings), task="solve x**2 - 4 = 0", data="x**2 - 4 = 0")
    assert result.ok
    assert result.result["solutions"] == ["-2", "2"]


def test_plot_compute(settings: BrainSettings) -> None:
    result = _run(_engine(settings), task="plot this data", data=[1, 2, 3, 4, 5])
    assert result.ok
    assert result.result["kind"] == "line"
    assert len(result.result["png_base64"]) > 100


def test_ml_compute(settings: BrainSettings) -> None:
    result = _run(_engine(settings), task="train a model to predict", data=SAMPLE_ROWS)
    assert result.ok
    assert result.capability == "ml"
    assert result.result["task"] == "classification"
    assert "accuracy" in result.result["metrics"]


def test_gradient_boost_compute(settings: BrainSettings) -> None:
    from companion.tools.common import module_available

    if not module_available("xgboost") and not module_available("lightgbm"):
        pytest.skip("neither xgboost nor lightgbm installed")
    result = _run(_engine(settings), task="train an xgboost model", data=SAMPLE_ROWS)
    assert result.ok
    assert result.result["task"] == "classification"


def test_simulation_compute(settings: BrainSettings) -> None:
    result = _run(_engine(settings), task="simulate a random walk", params={"steps": 20})
    assert result.ok
    assert result.capability == "simulation"
    assert len(result.result["series"]) == 20


def test_graph_compute(settings: BrainSettings) -> None:
    result = _run(
        _engine(settings),
        task="analyze this graph",
        data={"edges": [["A", "B"], ["B", "C"], ["C", "A"]]},
    )
    assert result.ok
    assert result.result["nodes"] == 3
    assert result.result["edges"] == 3


def test_vision_compute(settings: BrainSettings) -> None:
    result = _run(_engine(settings), task="describe this image", image_base64=_tiny_png())
    assert result.ok
    assert result.capability == "vision"
    assert result.result["width"] == 64
    assert result.result["height"] == 48


def test_nlp_compute(settings: BrainSettings) -> None:
    result = _run(
        _engine(settings),
        task="extract entities",
        data="Relay AI was founded in San Francisco to build automation tools.",
    )
    assert result.ok
    assert result.capability == "nlp"
    assert result.result["tokens"] >= 5


def test_jax_compute(settings: BrainSettings) -> None:
    from companion.tools.common import module_available

    if not module_available("jax"):
        pytest.skip("jax not installed")
    result = _run(_engine(settings), task="compute the gradient with jax")
    assert result.ok
    assert result.result["mode"] == "autodiff"


def test_torch_deep_learning_compute(settings: BrainSettings) -> None:
    from companion.tools.common import module_available

    if not module_available("torch"):
        pytest.skip("torch not installed")
    rows = [{"a": float(i), "b": float(i * 2), "y": float(i)} for i in range(20)]
    result = _run(_engine(settings), task="train a neural network", data=rows, params={"epochs": 20})
    assert result.ok
    assert result.result["engine"] == "pytorch"


def test_missing_data_returns_clean_error(settings: BrainSettings) -> None:
    result = _run(_engine(settings), task="analyze this csv")
    assert not result.ok
    assert "No data" in result.error


def test_unknown_capability_returns_clean_error(settings: BrainSettings) -> None:
    result = _run(_engine(settings), task="x", capability="nope")
    assert not result.ok
    assert "Unknown capability" in result.summary


# ── HTTP surface ────────────────────────────────────────────────────────

def _client(settings: BrainSettings) -> TestClient:
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app, headers={"Authorization": "Bearer test-only"})


def test_capabilities_endpoint(settings: BrainSettings) -> None:
    resp = _client(settings).get("/api/brain/capabilities")
    assert resp.status_code == 200
    ids = {c["id"] for c in resp.json()}
    assert "data-analysis" in ids and "vision" in ids and "deep-learning" in ids


def test_compute_endpoint(settings: BrainSettings) -> None:
    resp = _client(settings).post(
        "/api/brain/compute", json={"task": "calculate the mean", "data": [4, 8, 15, 16, 23, 42]}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["capability"] == "math"
    assert body["result"]["stats"]["mean"] == 18.0
