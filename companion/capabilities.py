"""Capability engine — Relay's computational toolbox + automatic selection.

This is the "Relay Capability → Available Tools → Required Framework → Execute
→ Return result" layer. Every framework the project ships is registered as a
capability with a lazy tool runner, so:

  - the catalog is discoverable (`GET /api/brain/capabilities`)
  - a task is auto-mapped to the best framework (`POST /api/brain/compute`)
  - nothing heavy is imported at startup or for unrelated requests
  - adding a future framework = one tool module + one registry entry

Selection scoring combines task keywords with data-shape hints so "analyze
this CSV" lands on Pandas while "analyze this image" lands on OpenCV.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .schemas import CapabilityInfo, ComputeRequest, ComputeResult
from .tools import (
    agents,
    ai,
    data,
    diffusion,
    face_search,
    graph,
    inference,
    ml,
    numeric,
    nlp,
    symbolic,
    training,
    vectors,
    vendor,
    vision,
    wheels,
)
from .tools.common import CapabilityUnavailable, module_available

# Display names for availability reporting.
LIBRARY_NAMES: dict[str, str] = {
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
    "qdrant_client": "Qdrant",
    "pymilvus": "Milvus",
    "vllm": "vLLM",
    "llama_cpp": "llama.cpp",
    "onnxruntime": "ONNX Runtime",
    "tensorrt": "NVIDIA TensorRT",
    "tensorrt_llm": "TensorRT-LLM",
    "triton": "Triton",
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

ToolFn = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class Capability:
    """One capability: an id, the frameworks it needs, and a lazy runner."""

    id: str
    label: str
    description: str
    libraries: list[str]
    keywords: tuple[str, ...]
    tool: ToolFn
    examples: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        return all(module_available(lib) for lib in self.libraries)

    @property
    def available_libraries(self) -> list[str]:
        return [lib for lib in self.libraries if module_available(lib)]

    @property
    def missing_libraries(self) -> list[str]:
        return [lib for lib in self.libraries if not module_available(lib)]


# ── the catalog ──────────────────────────────────────────────────────────

CAPABILITIES: list[Capability] = [
    Capability(
        id="math",
        label="Numerical computing",
        description="Statistics, linear algebra, transforms and math over numeric data.",
        libraries=["numpy", "scipy"],
        keywords=(
            "calculate", "compute", "statistics", "average", "mean", "median", "std",
            "variance", "matrix", "vector", "linear algebra", "fft", "fourier",
            "determinant", "eigen", "probability", "percentile", "numerical",
            "linear system", "system of equations",
        ),
        tool=numeric.run_numeric,
        examples=("mean, median, std of a list", "solve a linear system", "fft of a signal"),
    ),
    Capability(
        id="data-analysis",
        label="Data analysis",
        description="Profile, clean and summarize tabular data (CSV / rows).",
        libraries=["pandas", "numpy"],
        keywords=(
            "csv", "analyze", "dataframe", "dataset", "table", "spreadsheet",
            "excel", "aggregate", "clean the data", "summarize this data",
            "data profiling", "correlation",
        ),
        tool=data.run_data_analysis,
        examples=("analyze this CSV", "profile this dataset", "show correlations"),
    ),
    Capability(
        id="plot",
        label="Charts & visualization",
        description="Render line, bar, scatter or histogram charts from data.",
        libraries=["matplotlib", "numpy"],
        keywords=(
            "plot", "chart", "visualize", "visualization", "histogram", "scatter",
            "graph this", "barchart",
        ),
        tool=data.run_plot,
        examples=("plot this data as a line chart", "histogram of this column"),
    ),
    Capability(
        id="ml",
        label="Classic machine learning",
        description="Train and evaluate Scikit-learn models (classify/regress/cluster/anomaly).",
        libraries=["sklearn", "numpy"],
        keywords=(
            "train a model", "predict", "classify", "regression", "machine learning",
            "fit a model", "clustering", "anomaly", "classification", "random forest",
        ),
        tool=ml.run_ml,
        examples=("train a model to predict this", "cluster these rows", "find anomalies"),
    ),
    Capability(
        id="gradient-boost",
        label="Gradient boosting",
        description="High-performance boosted trees (XGBoost or LightGBM).",
        libraries=["xgboost", "lightgbm"],
        keywords=(
            "gradient boosting", "xgboost", "lightgbm", "boosted tree", "boosting",
            "boosted model",
        ),
        tool=ml.run_gradient_boost,
        examples=("train an XGBoost model", "gradient boosting for this dataset"),
    ),
    Capability(
        id="symbolic",
        label="Symbolic math",
        description="Solve equations, algebra, calculus and simplification with SymPy.",
        libraries=["sympy"],
        keywords=(
            "equation", "solve", "solve for", "algebra", "derive", "differentiate",
            "integrate", "simplify", "factor", "expand", "calculus", "symbolic",
            "solve the equation",
        ),
        tool=symbolic.run_symbolic,
        examples=("solve x^2 - 4 = 0", "differentiate x^3 + 2x", "integrate sin(x)"),
    ),
    Capability(
        id="simulation",
        label="Simulation",
        description="Numerical simulations: random walks, Monte Carlo, growth, oscillators.",
        libraries=["numpy", "scipy"],
        keywords=(
            "simulate", "simulation", "random walk", "monte carlo", "growth model",
            "dynamics", "model the",
        ),
        tool=numeric.run_simulation,
        examples=("simulate a random walk", "estimate pi with monte carlo"),
    ),
    Capability(
        id="graph",
        label="Graphs & networks",
        description="Graph structure, paths, components, centrality and cycles with NetworkX.",
        libraries=["networkx"],
        keywords=(
            "shortest path", "graph", "graph theory", "nodes and edges",
            "connected components", "graph analysis", "network analysis", "centrality",
        ),
        tool=graph.run_graph,
        examples=("shortest path between A and B", "analyze this graph"),
    ),
    Capability(
        id="vision",
        label="Computer vision",
        description="Image processing, features and local vision models via OpenCV + Pillow.",
        libraries=["cv2", "PIL"],
        keywords=(
            "image", "photo", "picture", "ocr", "face detection", "object detection",
            "camera", "computer vision", "resize image", "image processing", "analyze the image",
        ),
        tool=vision.run_vision,
        examples=("describe this image", "detect edges in this photo", "resize the image"),
    ),
    Capability(
        id="face-search",
        label="Face / reverse-image search",
        description=(
            "Upload a photo of a person (or any image) and search the public "
            "internet for matching pages: social profiles across LinkedIn, "
            "Instagram, Facebook, X, TikTok and more, plus news, blogs and "
            "company pages. Providers: SerpAPI engines (Google Lens, Yandex, "
            "Bing, Google Images), TinEye, and keyless name-based search. "
            "Face verification of candidate thumbnails runs locally when the "
            "vision stack is present."
        ),
        libraries=["httpx"],  # network runtime required; face models remain optional
        keywords=(
            "face search", "face recognition search", "reverse image search",
            "find this person", "find profiles", "identify person", "who is this person",
            "social profiles from photo", "search by photo", "image search",
            "person search", "find social media", "osint face",
        ),
        tool=face_search.run_face_search,
        examples=(
            "find this person's social profiles from a photo",
            "reverse image search this picture",
            "where does this photo appear online",
        ),
    ),
    Capability(
        id="nlp",
        label="NLP & text analysis",
        description="Tokenization, entities, POS and keywords via spaCy / NLTK.",
        libraries=["spacy", "nltk"],
        keywords=(
            "tokens", "entities", "nlp", "parts of speech", "language analysis",
            "text analysis", "summarize text", "keywords in the text", "lemmatize",
        ),
        tool=nlp.run_nlp,
        examples=("extract entities from this text", "analyze this text"),
    ),
    Capability(
        id="deep-learning",
        label="Deep learning (PyTorch)",
        description="Neural networks and autograd with PyTorch.",
        libraries=["torch"],
        keywords=(
            "neural network", "pytorch", "deep learning", "mlp", "train a network",
            "autograd",
        ),
        tool=ai.run_deep_learning,
        examples=("train a neural network on this data", "compute the gradient"),
    ),
    Capability(
        id="local-llm",
        label="Local open-source models",
        description="Run or embed with open-source Hugging Face models locally.",
        libraries=["transformers", "torch"],
        keywords=(
            "hugging face", "local model", "open source model", "run a model",
            "model inference", "distilgpt", "local llm", "text embedding",
        ),
        tool=ai.run_local_llm,
        examples=("generate text with a local model", "embed this text"),
    ),
    Capability(
        id="llm-gateway",
        label="Unified LLM gateway",
        description="Call any LLM provider through one interface (LiteLLM).",
        libraries=["litellm"],
        keywords=(
            "litellm", "unified llm", "switch provider", "use gemini", "use openai",
            "use claude", "provider routing", "model routing", "call the llm",
        ),
        tool=ai.run_llm_gateway,
        examples=("use Gemini through the unified gateway", "call claude"),
    ),
    Capability(
        id="rag",
        label="Document RAG",
        description="Index documents and retrieve answers with LlamaIndex.",
        libraries=["llama_index"],
        keywords=(
            "documents", "knowledge base", "index my documents", "answer from",
            "retrieval", "rag", "search my documents", "document search",
        ),
        tool=ai.run_rag,
        examples=("search my documents and answer", "index these documents"),
    ),
    Capability(
        id="tensorflow",
        label="TensorFlow / Keras",
        description="Train small Keras models with the TensorFlow backend.",
        libraries=["tensorflow", "keras"],
        keywords=("tensorflow", "keras", "tf model"),
        tool=ai.run_tensorflow,
        examples=("train a keras model"),
    ),
    Capability(
        id="jax",
        label="JAX",
        description="High-performance math, autodiff and JIT with JAX.",
        libraries=["jax"],
        keywords=("jax", "autodiff", "automatic differentiation", "jit"),
        tool=ai.run_jax,
        examples=("compute the gradient with jax", "jit a computation"),
    ),
    Capability(
        id="vllm",
        label="vLLM inference",
        description="High-throughput LLM generation with vLLM (GPU).",
        libraries=["vllm"],
        keywords=("vllm", "high throughput inference", "serve the model"),
        tool=inference.run_vllm,
        examples=("generate with vllm", "vllm health check"),
    ),
    Capability(
        id="ollama",
        label="Ollama local models",
        description="Generate or embed through a local Ollama server (HTTP; transport is httpx).",
        libraries=["httpx"],
        keywords=("ollama", "local model server", "run a local llama"),
        tool=inference.run_ollama,
        examples=("list ollama models", "generate with ollama", "embed text with ollama"),
    ),
    Capability(
        id="llamacpp",
        label="llama.cpp (GGUF)",
        description="Run GGUF models locally with llama-cpp-python (CPU/GPU).",
        libraries=["llama_cpp"],
        keywords=("llama.cpp", "gguf", "llama cpp", "ggml model"),
        tool=inference.run_llamacpp,
        examples=("run a gguf model with llama.cpp", "llama cpp health"),
    ),
    Capability(
        id="onnx",
        label="ONNX Runtime",
        description="Cross-platform ONNX inference and provider introspection.",
        libraries=["onnxruntime"],
        keywords=("onnx", "onnx runtime", "open neural network exchange"),
        tool=inference.run_onnx,
        examples=("list onnx providers", "run an onnx model"),
    ),
    Capability(
        id="tgi",
        label="Hugging Face TGI",
        description="Generate through a Hugging Face Text Generation Inference server (HTTP; transport is httpx).",
        libraries=["httpx"],
        keywords=("tgi", "text generation inference", "hf inference server"),
        tool=inference.run_tgi,
        examples=("call a tgi server", "tgi generate"),
    ),
    Capability(
        id="tensorrt",
        label="NVIDIA TensorRT",
        description="TensorRT GPU engine introspection (GPU only).",
        libraries=["tensorrt"],
        keywords=("tensorrt", "nvidia trt"),
        tool=inference.run_tensorrt,
        examples=("tensorrt health", "check tensorrt version"),
    ),
    Capability(
        id="tensorrt-llm",
        label="TensorRT-LLM",
        description="NVIDIA TensorRT-LLM availability + GPU check.",
        libraries=["tensorrt_llm"],
        keywords=("tensorrt-llm", "tensorrt llm"),
        tool=inference.run_tensorrt_llm,
        examples=("tensorrt-llm check"),
    ),
    Capability(
        id="triton",
        label="Triton GPU kernels",
        description="JIT-compile and run Triton GPU kernels (needs CUDA).",
        libraries=["triton"],
        keywords=("triton kernel", "gpu kernel", "triton compile"),
        tool=inference.run_triton,
        examples=("compile a triton kernel", "triton probe"),
    ),
    Capability(
        id="ggml",
        label="ggml backend",
        description="Probe the ggml backend (practical Python path is llama.cpp bindings).",
        libraries=["llama_cpp"],
        keywords=("ggml", "ggml backend"),
        tool=inference.run_ggml,
        examples=("ggml probe"),
    ),
    Capability(
        id="diffusion",
        label="Text-to-image (diffusers)",
        description="Generate images from prompts with Hugging Face diffusers (GPU).",
        libraries=["diffusers"],
        keywords=(
            "generate an image", "text to image", "diffusers", "stable diffusion",
            "image generation", "create a picture", "draw a", "render an image",
        ),
        tool=diffusion.run_diffusion,
        examples=("generate an image of a sunset city", "stable diffusion prompt"),
    ),
    Capability(
        id="timm",
        label="Vision models (timm)",
        description="Catalogue and run vision models with pytorch-image-models (GPU).",
        libraries=["timm"],
        keywords=("timm", "pytorch-image-models", "vision model", "image features"),
        tool=diffusion.run_timm,
        examples=("list timm models", "run resnet features"),
    ),
    Capability(
        id="accelerate",
        label="Hugging Face Accelerate",
        description="Device placement and model dispatch with accelerate.",
        libraries=["accelerate"],
        keywords=("accelerate", "device map", "model dispatch", "offload"),
        tool=diffusion.run_accelerate,
        examples=("accelerate device state", "offload a model with accelerate"),
    ),
    Capability(
        id="deepspeed",
        label="DeepSpeed",
        description="DeepSpeed health, CUDA check and training-config validation.",
        libraries=["deepspeed"],
        keywords=("deepspeed", "zero stage", "distributed training"),
        tool=training.run_deepspeed,
        examples=("deepspeed health", "validate a deepspeed config"),
    ),
    Capability(
        id="megatron",
        label="Megatron-LM",
        description="Megatron-LM availability probe (GPU research framework).",
        libraries=["megatron"],
        keywords=("megatron", "megatron-lm"),
        tool=training.run_megatron,
        examples=("megatron probe"),
    ),
    Capability(
        id="axolotl",
        label="axolotl fine-tuning",
        description="Generate validated axolotl fine-tuning configs for the CLI.",
        libraries=["axolotl"],
        keywords=("axolotl", "fine tuning config", "sft config"),
        tool=training.run_axolotl,
        examples=("build an axolotl config", "fine-tune with axolotl"),
    ),
    Capability(
        id="bitsandbytes",
        label="bitsandbytes quantization",
        description="8-bit quantization layers with bitsandbytes (GPU).",
        libraries=["bitsandbytes"],
        keywords=("bitsandbytes", "8-bit", "quantization", "4-bit", "bnb"),
        tool=training.run_bitsandbytes,
        examples=("bitsandbytes health", "build an 8-bit layer"),
    ),
    Capability(
        id="unsloth",
        label="unsloth fine-tuning",
        description="unsloth fast fine-tuning availability and model loading (GPU).",
        libraries=["unsloth"],
        keywords=("unsloth", "fast fine tuning", "4-bit load"),
        tool=training.run_unsloth,
        examples=("unsloth health", "load a model with unsloth"),
    ),
    Capability(
        id="langchain",
        label="LangChain",
        description="Build and run LangChain chains over the configured LLM.",
        libraries=["langchain"],
        keywords=("langchain", "chain", "llm pipeline", "lc"),
        tool=agents.run_langchain,
        examples=("build a langchain chain", "langchain providers"),
    ),
    Capability(
        id="crewai",
        label="crewAI agents",
        description="Assemble and run a crewAI crew (agents + task).",
        libraries=["crewai"],
        keywords=("crewai", "crew", "agent team", "multi agent task"),
        tool=agents.run_crewai,
        examples=("run a crewai crew", "create an agent team"),
    ),
    Capability(
        id="autogen",
        label="Microsoft autogen",
        description="Two-agent autogen conversations.",
        libraries=["autogen"],
        keywords=("autogen", "microsoft autogen", "multi agent conversation"),
        tool=agents.run_autogen,
        examples=("autogen conversation", "run autogen agents"),
    ),
    Capability(
        id="langflow",
        label="Langflow",
        description="List and run flows on a Langflow instance (HTTP; transport is httpx).",
        libraries=["httpx"],
        keywords=("langflow", "low code flow", "flow platform"),
        tool=agents.run_langflow,
        examples=("list langflow flows", "run a langflow flow"),
    ),
    Capability(
        id="qdrant",
        label="Qdrant vector search",
        description="List collections and vector-search with qdrant-client.",
        libraries=["qdrant_client"],
        keywords=("qdrant", "vector database", "vector search", "semantic search"),
        tool=vectors.run_qdrant,
        examples=("list qdrant collections", "search a qdrant collection"),
    ),
    Capability(
        id="milvus",
        label="Milvus vector database",
        description="List collections and vector-search with pymilvus.",
        libraries=["pymilvus"],
        keywords=("milvus", "milvus search", "vector db"),
        tool=vectors.run_milvus,
        examples=("list milvus collections", "search milvus"),
    ),
    Capability(
        id="vendor-source",
        label="Vendored framework source",
        description=(
            "Inventory the AI/ML framework source shipped in-repo (importable "
            "packages + compiled-giant source archives + bundled wheels). "
            "Always available."
        ),
        libraries=[],
        keywords=(
            "vendored framework", "local framework source", "source archive",
            "which frameworks", "local models source", "offline framework",
            "framework inventory",
        ),
        tool=vendor.run_vendor_source,
        examples=(
            "which AI frameworks does relay have source for",
            "list the vendored framework archives",
        ),
    ),
    Capability(
        id="wheel-install",
        label="Bundled wheel provisioning",
        description=(
            "Inspect locally stored pre-built wheels (PyTorch CUDA bundle, "
            "TensorFlow, ONNX Runtime GPU) and — when the host enables it — "
            "install one from the registry. dry_run mode validates without "
            "touching the environment. Always available."
        ),
        libraries=[],
        keywords=(
            "install torch", "install a wheel", "bundled wheel", "cuda install",
            "provision", "wheel registry", "install onnxruntime", "install tensorflow",
            "what wheels are stored", "self provision",
        ),
        tool=wheels.run_wheel_install,
        examples=(
            "which bundled wheels are stored",
            "install the torch CUDA wheel from the bundle",
            "status of stored wheels",
        ),
    ),
]

_BY_ID: dict[str, Capability] = {cap.id: cap for cap in CAPABILITIES}


def list_capabilities() -> list[CapabilityInfo]:
    """The full catalog with live availability, for `GET /api/brain/capabilities`."""
    return [
        CapabilityInfo(
            id=cap.id,
            label=cap.label,
            description=cap.description,
            available=cap.available,
            libraries=[LIBRARY_NAMES.get(lib, lib) for lib in cap.libraries],
            available_libraries=[LIBRARY_NAMES.get(lib, lib) for lib in cap.available_libraries],
            missing_libraries=[LIBRARY_NAMES.get(lib, lib) for lib in cap.missing_libraries],
            examples=list(cap.examples),
        )
        for cap in CAPABILITIES
    ]


# ── automatic selection ──────────────────────────────────────────────────

# Generic action words that never route a task on their own — "analyze this
# graph" must reach NetworkX, not Pandas; "compute the gradient with jax"
# must reach JAX, not the numerics fallback. Phrased keywords (e.g. "analyze
# the image", "use gemini") still count.
_CONTEXT_WORDS = {
    "analyze", "compute", "calculate", "look", "show", "please", "help",
    "examine", "review", "inspect", "can", "you", "for", "me", "the", "this",
    "make", "do", "create", "run", "use", "get", "at", "and", "with", "a",
}


def resolve_capability(task: str, capability_id: str = "", data: Any = None) -> tuple[Capability, int]:
    """Pick the best capability for a task, or the explicit one.

    Returns (capability, match_score). An explicit `capability_id` always
    wins. Otherwise keyword hits on the task + data-shape hints decide.
    """
    if capability_id:
        cap = _BY_ID.get(capability_id)
        if cap is None:
            raise ValueError(
                f"Unknown capability '{capability_id}'. Known: {', '.join(sorted(_BY_ID))}"
            )
        return cap, 1

    text = (task or "").lower()
    best: Optional[Capability] = None
    best_score = 0
    best_specificity = 0
    for cap in CAPABILITIES:
        score = 0
        max_keyword_len = 0
        for keyword in cap.keywords:
            if keyword.strip().lower() in _CONTEXT_WORDS:
                continue
            if _contains_keyword(text, keyword):
                score += 1
                max_keyword_len = max(max_keyword_len, len(keyword))
        # Prefer the most specific phrasing on ties: "generate an image" (18
        # chars) must beat the bare "image" keyword on vision so text-to-image
        # tasks reach diffusers instead of image analysis.
        if score > best_score or (score == best_score and score > 0 and max_keyword_len > best_specificity):
            best, best_score, best_specificity = cap, score, max_keyword_len

    # Any direct keyword match wins — trust the user's words over data shape
    # ("plot this data" must reach Matplotlib, not the numerics fallback).
    if best is not None and best_score >= 1:
        return best, best_score

    # No keyword matched: fall back to data shape (CSV → Pandas, numbers →
    # NumPy, text → NLP, image → vision).
    return _resolve_by_shape(data)


def _contains_keyword(text: str, keyword: str) -> bool:
    if " " in keyword:
        return keyword in text
    return bool(re.search(rf"\b{re.escape(keyword)}\b", text))


def _resolve_by_shape(data: Any) -> tuple[Capability, int]:
    image = None
    if isinstance(data, dict):
        image = data.get("image_base64")
    if image:
        return _BY_ID["vision"], 3
    if isinstance(data, str):
        stripped = data.strip()
        if "\n" in stripped or "," in stripped:
            return _BY_ID["data-analysis"], 3
        if len(data) > 20:
            return _BY_ID["nlp"], 2
        return _BY_ID["math"], 1
    if isinstance(data, list):
        if data and all(isinstance(d, (dict, list, tuple)) for d in data):
            return _BY_ID["data-analysis"], 3
        if data and all(isinstance(d, (int, float)) for d in data):
            return _BY_ID["math"], 2
        if data and all(isinstance(d, str) for d in data):
            return _BY_ID["nlp"], 2
    if isinstance(data, dict) and "rows" in data:
        return _BY_ID["data-analysis"], 3
    return _BY_ID["math"], 1


# ── the engine ───────────────────────────────────────────────────────────

class CapabilityEngine:
    """Runs tasks through the right framework, lazily and safely."""

    def __init__(self, settings: Any = None) -> None:
        self.settings = settings

    def catalog(self) -> list[CapabilityInfo]:
        return list_capabilities()

    async def run(self, request: ComputeRequest) -> ComputeResult:
        try:
            cap, score = resolve_capability(request.task, request.capability, request.data)
        except ValueError as exc:
            return ComputeResult(
                capability=request.capability or request.task[:40],
                ok=False,
                error=str(exc),
                summary=f"Unknown capability: {exc}",
            )

        if not cap.available:
            return ComputeResult(
                capability=cap.id,
                ok=False,
                error=(
                    f"The {cap.label} capability needs {', '.join(cap.missing_libraries)} "
                    f"which are not installed."
                ),
                summary=(
                    f"The {cap.label} capability needs {', '.join(cap.missing_libraries)}. "
                    f"Install them with `pip install -r requirements.companion-ai.txt`."
                ),
            )

        payload = {
            "task": request.task,
            "capability": cap.id,
            "data": request.data,
            "params": request.params or {},
            "image_base64": request.image_base64,
            "_settings": self.settings,
        }
        try:
            outcome = await asyncio.to_thread(cap.tool, payload)
        except CapabilityUnavailable as exc:
            return ComputeResult(
                capability=cap.id,
                ok=False,
                error=str(exc),
                summary=str(exc),
            )
        except Exception as exc:  # noqa: BLE001 - tools surface errors as results
            return ComputeResult(
                capability=cap.id,
                ok=False,
                error=str(exc)[:500],
                summary=f"Computation failed: {exc}"[:500],
            )

        libraries_used = [LIBRARY_NAMES.get(lib, lib) for lib in outcome.get("libraries_used", [])]
        result = outcome.get("result")
        if result is None or result is False:
            return ComputeResult(
                capability=cap.id,
                ok=False,
                error=outcome.get("summary", "No result"),
                summary=str(outcome.get("summary", "No result")),
            )

        return ComputeResult(
            capability=cap.id,
            ok=True,
            result=result,
            summary=outcome.get("summary", "Done."),
            libraries_used=libraries_used,
        )
