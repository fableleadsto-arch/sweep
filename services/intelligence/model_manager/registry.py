"""Model registry — tracks all installed models, their status, and metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any


class ModelStatus(str, Enum):
    NOT_DOWNLOADED = "not_downloaded"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    LOADED = "loaded"
    ERROR = "error"
    UNAVAILABLE = "unavailable"


@dataclass
class ModelEntry:
    name: str
    category: str  # nlp, embeddings, vision, audio, face, documents, detection
    task: str  # what it does
    provider: str  # huggingface, ultralytics, opencv, etc.
    checkpoint: str  # HF repo or model ID
    local_path: str  # relative to models/
    license: str = ""
    download_size_mb: float = 0
    memory_required_mb: float = 0
    status: ModelStatus = ModelStatus.NOT_DOWNLOADED
    lazy_load: bool = True
    optional: bool = False
    requires_gpu: bool = False
    requires_auth: bool = False
    verified: bool = False
    error: str = ""
    modality: str = ""  # text, image, audio, video, document, multimodal


# ── Canonical registry ────────────────────────────────────────────────────
REGISTRY: list[ModelEntry] = [
    # NLP
    ModelEntry(
        name="bert-base-uncased", category="nlp", task="language_understanding",
        provider="huggingface", checkpoint="bert-base-uncased",
        local_path="models/nlp/bert-base-uncased", license="Apache-2.0",
        download_size_mb=440, memory_required_mb=700, modality="text",
    ),
    ModelEntry(
        name="deberta-v3-base", category="nlp", task="intent_classification",
        provider="huggingface", checkpoint="microsoft/deberta-v3-base",
        local_path="models/nlp/deberta-v3-base", license="MIT",
        download_size_mb=510, memory_required_mb=800, modality="text",
    ),
    ModelEntry(
        name="roberta-base", category="nlp", task="text_classification",
        provider="huggingface", checkpoint="FacebookAI/roberta-base",
        local_path="models/nlp/roberta-base", license="MIT",
        download_size_mb=500, memory_required_mb=700, modality="text",
    ),
    ModelEntry(
        name="xlm-roberta-base", category="nlp", task="multilingual_understanding",
        provider="huggingface", checkpoint="FacebookAI/xlm-roberta-base",
        local_path="models/nlp/xlm-roberta-base", license="MIT",
        download_size_mb=1100, memory_required_mb=1200, modality="text",
    ),
    ModelEntry(
        name="gliner-base-v2.1", category="nlp", task="entity_extraction",
        provider="huggingface", checkpoint="urchade/gliner_medium-v2.1",
        local_path="models/nlp/gliner_base-v2.1", license="Apache-2.0",
        download_size_mb=500, memory_required_mb=700, modality="text",
    ),

    # Embeddings
    ModelEntry(
        name="all-MiniLM-L6-v2", category="embeddings", task="text_embedding",
        provider="sentence-transformers", checkpoint="sentence-transformers/all-MiniLM-L6-v2",
        local_path="models/embeddings/all-MiniLM-L6-v2", license="Apache-2.0",
        download_size_mb=90, memory_required_mb=130, modality="text", lazy_load=False,
    ),
    ModelEntry(
        name="all-mpnet-base-v2", category="embeddings", task="text_embedding",
        provider="sentence-transformers", checkpoint="sentence-transformers/all-mpnet-base-v2",
        local_path="models/embeddings/all-mpnet-base-v2", license="Apache-2.0",
        download_size_mb=420, memory_required_mb=500, modality="text",
    ),

    # Vision — CLIP
    ModelEntry(
        name="clip-vit-b32", category="vision", task="multimodal_embedding",
        provider="open_clip", checkpoint="ViT-B-32::openai",
        local_path="models/vision/clip-vit-b32", license="MIT",
        download_size_mb=350, memory_required_mb=500, modality="multimodal",
    ),
    ModelEntry(
        name="clip-vit-l14", category="vision", task="multimodal_embedding",
        provider="open_clip", checkpoint="ViT-L-14::openai",
        local_path="models/vision/clip-vit-l14", license="MIT",
        download_size_mb=1700, memory_required_mb=2000, modality="multimodal",
        optional=True,
    ),

    # Vision — YOLO
    ModelEntry(
        name="yolo-v8n", category="detection", task="object_detection",
        provider="ultralytics", checkpoint="yolov8n.pt",
        local_path="models/detection/yolo", license="AGPL-3.0",
        download_size_mb=6, memory_required_mb=50, modality="image",
    ),

    # Face
    ModelEntry(
        name="insightface", category="face", task="face_detection_embedding",
        provider="insightface", checkpoint="buffalo_l",
        local_path="models/face/insightface", license="MIT",
        download_size_mb=300, memory_required_mb=400, modality="image",
    ),

    # Audio / Speech
    ModelEntry(
        name="whisper-base", category="speech", task="speech_to_text",
        provider="openai-whisper", checkpoint="base",
        local_path="models/speech", license="MIT",
        download_size_mb=150, memory_required_mb=300, modality="audio",
    ),
    ModelEntry(
        name="wav2vec2-base", category="audio", task="audio_representation",
        provider="huggingface", checkpoint="facebook/wav2vec2-base",
        local_path="models/audio/wav2vec2", license="Apache-2.0",
        download_size_mb=360, memory_required_mb=500, modality="audio",
    ),
    ModelEntry(
        name="pyannote-segmentation", category="audio", task="speaker_diarization",
        provider="pyannote", checkpoint="pyannote/segmentation-3.0",
        local_path="models/audio/pyannote", license="MIT",
        download_size_mb=10, memory_required_mb=200, modality="audio",
        requires_auth=True,
    ),

    # Documents
    ModelEntry(
        name="layoutlm-base", category="documents", task="document_understanding",
        provider="huggingface", checkpoint="microsoft/layoutlm-base-uncased",
        local_path="models/documents/layoutlm", license="MIT",
        download_size_mb=440, memory_required_mb=700, modality="document",
    ),
    ModelEntry(
        name="layoutlmv3-base", category="documents", task="document_ocr_layout",
        provider="huggingface", checkpoint="microsoft/layoutlmv3-base",
        local_path="models/documents/layoutlmv3-base", license="MIT",
        download_size_mb=500, memory_required_mb=800, modality="document",
    ),

    # Vision — OpenCV (no model download)
    ModelEntry(
        name="opencv", category="vision", task="image_processing",
        provider="opencv", checkpoint="",
        local_path="", license="Apache-2.0",
        download_size_mb=0, memory_required_mb=0, modality="image",
        lazy_load=False,
    ),

    # Vision — Tesseract (system binary)
    ModelEntry(
        name="tesseract", category="documents", task="ocr",
        provider="tesseract", checkpoint="",
        local_path="", license="Apache-2.0",
        download_size_mb=0, memory_required_mb=0, modality="image",
        lazy_load=False,
    ),
]


class ModelRegistry:
    """In-memory model registry with persistence."""

    def __init__(self, models_root: str = "models"):
        self.models_root = Path(models_root)
        self._entries: dict[str, ModelEntry] = {}
        for entry in REGISTRY:
            self._entries[entry.name] = replace(entry)
        self._load_status()
        self.sync_from_disk()

    def sync_from_disk(self):
        """Detect models actually present on disk and update statuses."""
        for entry in self._entries.values():
            if entry.local_path == "":
                # package/system-provided — detect availability by import/binary
                continue
            path = Path(entry.local_path)
            if not path.exists():
                if entry.status in (ModelStatus.DOWNLOADED, ModelStatus.LOADED):
                    entry.status = ModelStatus.NOT_DOWNLOADED
                continue
            files = list(path.rglob("*"))
            has_weights = any(
                f.suffix in (".bin", ".safetensors", ".pt", ".onnx") for f in files
            )
            has_config = any(f.name == "config.json" or f.name == "modules.json" for f in files)
            if has_weights or (has_config and entry.provider == "sentence-transformers"):
                if entry.status == ModelStatus.NOT_DOWNLOADED:
                    entry.status = ModelStatus.DOWNLOADED

    def _status_path(self) -> Path:
        return self.models_root / "registry.json"

    def _load_status(self):
        """Load download/verification status from disk."""
        path = self._status_path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            for name, info in data.items():
                if name in self._entries:
                    self._entries[name].status = ModelStatus(info.get("status", "not_downloaded"))
                    self._entries[name].verified = info.get("verified", False)
                    self._entries[name].error = info.get("error", "")
        except (json.JSONDecodeError, KeyError):
            pass

    def save_status(self):
        """Persist download/verification status to disk."""
        self.models_root.mkdir(parents=True, exist_ok=True)
        data = {}
        for name, entry in self._entries.items():
            data[name] = {
                "status": entry.status.value,
                "verified": entry.verified,
                "error": entry.error,
                "local_path": entry.local_path,
            }
        self._status_path().write_text(json.dumps(data, indent=2))

    def get(self, name: str) -> ModelEntry | None:
        return self._entries.get(name)

    def all(self) -> list[ModelEntry]:
        return list(self._entries.values())

    def by_category(self, category: str) -> list[ModelEntry]:
        return [e for e in self._entries.values() if e.category == category]

    def by_task(self, task: str) -> list[ModelEntry]:
        return [e for e in self._entries.values() if e.task == task]

    def by_modality(self, modality: str) -> list[ModelEntry]:
        return [e for e in self._entries.values() if e.modality == modality]

    def installed(self) -> list[ModelEntry]:
        return [e for e in self._entries.values()
                if e.status in (ModelStatus.DOWNLOADED, ModelStatus.LOADED)]

    def missing(self) -> list[ModelEntry]:
        return [e for e in self._entries.values()
                if e.status == ModelStatus.NOT_DOWNLOADED and not e.optional]
