"""Model loader — lazy loading with device detection and error handling."""

from __future__ import annotations

from services.model_loading import serialized_model_load

import logging
import time
from pathlib import Path
from typing import Any

from .cache import ModelCache
from .device import DeviceInfo, detect_device, get_device_map, get_torch_device
from .registry import ModelEntry, ModelRegistry, ModelStatus

log = logging.getLogger("sweep.intelligence.loader")


class ModelLoader:
    """Lazy-loads models from local paths with device awareness."""

    def __init__(self, registry: ModelRegistry, device: DeviceInfo | None = None):
        self.registry = registry
        self.device = device or detect_device()
        self.cache = ModelCache(
            max_size=4 if self.device.ram_available_gb < 4 else 8,
            max_memory_mb=int(min(self.device.ram_available_gb * 0.5 * 1000, 4000)),
        )
        self._loaders: dict[str, Any] = {
            # NLP
            "bert-base-uncased": self._load_transformers_nlp,
            "deberta-v3-base": self._load_transformers_nlp,
            "roberta-base": self._load_transformers_nlp,
            "xlm-roberta-base": self._load_transformers_nlp,
            "gliner-base-v2.1": self._load_gliner,
            # Embeddings
            "all-MiniLM-L6-v2": self._load_sentence_transformer,
            "all-mpnet-base-v2": self._load_sentence_transformer,
            # Vision
            "clip-vit-b32": self._load_clip,
            "clip-vit-l14": self._load_clip,
            "yolo-v8n": self._load_yolo,
            # Face
            "insightface": self._load_insightface,
            # Audio
            "whisper-base": self._load_whisper,
            "wav2vec2-base": self._load_wav2vec2,
            "pyannote-segmentation": self._load_pyannote,
            # Documents
            "layoutlm-base": self._load_layoutlm,
            "layoutlmv3-base": self._load_layoutlmv3,
        }

    @serialized_model_load
    def load(self, name: str) -> Any:
        """Load a model by name. Returns cached if already loaded."""
        # Check cache first
        cached = self.cache.get(name)
        if cached is not None:
            return cached

        entry = self.registry.get(name)
        if entry is None:
            raise ValueError(f"Unknown model: {name}")

        if entry.status == ModelStatus.NOT_DOWNLOADED:
            raise RuntimeError(
                f"Model {name} not downloaded. Run: python services/intelligence/download_models.py"
            )

        # Check GPU requirements
        if entry.requires_gpu and self.device.device == "cpu":
            raise RuntimeError(
                f"Model {name} requires GPU but only CPU is available"
            )

        loader = self._loaders.get(name)
        if loader is None:
            raise RuntimeError(f"No loader registered for model: {name}")

        t0 = time.time()
        try:
            model = loader(entry)
            self.cache.put(name, model, size_mb=entry.memory_required_mb)
            entry.status = ModelStatus.LOADED
            self.registry.save_status()
            elapsed = time.time() - t0
            log.info(f"Loaded {name} in {elapsed:.1f}s on {self.device.device}")
            return model
        except Exception as e:
            entry.status = ModelStatus.ERROR
            entry.error = str(e)
            self.registry.save_status()
            log.error(f"Failed to load {name}: {e}")
            raise

    def unload(self, name: str) -> bool:
        removed = self.cache.remove(name)
        entry = self.registry.get(name)
        if entry and entry.status == ModelStatus.LOADED:
            entry.status = ModelStatus.DOWNLOADED
        return removed

    def loaded_models(self) -> list[str]:
        return self.cache.loaded_models

    def cache_stats(self) -> dict:
        return self.cache.stats()

    # ── Private loaders ───────────────────────────────────────────────────

    def _load_transformers_nlp(self, entry: ModelEntry) -> dict:
        from transformers import AutoModel, AutoTokenizer
        local = str(Path(entry.local_path))
        device_map = get_device_map(self.device)
        tokenizer = AutoTokenizer.from_pretrained(local, local_files_only=True)
        model = AutoModel.from_pretrained(local, local_files_only=True, device_map=device_map)
        return {"model": model, "tokenizer": tokenizer, "framework": "transformers"}

    def _load_gliner(self, entry: ModelEntry) -> dict:
        from gliner import GLiNER
        local = str(Path(entry.local_path))
        model = GLiNER.from_pretrained(local)
        return {"model": model, "framework": "gliner"}

    def _load_sentence_transformer(self, entry: ModelEntry) -> dict:
        from sentence_transformers import SentenceTransformer
        local = str(Path(entry.local_path))
        model = SentenceTransformer(local)  # local path, will use cache if files exist
        return {"model": model, "framework": "sentence_transformers"}

    def _load_clip(self, entry: ModelEntry) -> dict:
        import open_clip
        import torch
        parts = entry.checkpoint.split("::")
        model_name = parts[0]
        pretrained = parts[1] if len(parts) > 1 else "openai"
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device.device
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        return {
            "model": model, "preprocess": preprocess, "tokenizer": tokenizer,
            "framework": "open_clip",
        }

    def _load_yolo(self, entry: ModelEntry) -> dict:
        from ultralytics import YOLO
        model_path = str(Path(entry.local_path) / "yolov8n.pt")
        model = YOLO(model_path)
        return {"model": model, "framework": "ultralytics"}

    def _load_insightface(self, entry: ModelEntry) -> dict:
        import insightface
        app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            root=str(Path(entry.local_path)),  # expects <root>/models/buffalo_l/*.onnx
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        return {"model": app, "framework": "insightface"}

    def _load_whisper(self, entry: ModelEntry) -> dict:
        import whisper
        # whisper stores checkpoints as <root>/base.pt
        model = whisper.load_model(entry.checkpoint, download_root="models/speech")
        return {"model": model, "framework": "whisper"}

    def _load_wav2vec2(self, entry: ModelEntry) -> dict:
        from transformers import AutoModel, AutoProcessor
        local = str(Path(entry.local_path))
        processor = AutoProcessor.from_pretrained(local, local_files_only=True)
        model = AutoModel.from_pretrained(local, local_files_only=True)
        return {"model": model, "processor": processor, "framework": "transformers"}

    def _load_pyannote(self, entry: ModelEntry) -> dict:
        from pyannote.audio import Pipeline
        import os
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            raise RuntimeError(
                "pyannote.audio requires HF token. Set HF_TOKEN env var. "
                "Accept the model license at: https://huggingface.co/pyannote/segmentation-3.0"
            )
        pipeline = Pipeline.from_pretrained(
            "pyannote/segmentation-3.0",
            use_auth_token=token,
        )
        return {"model": pipeline, "framework": "pyannote"}

    def _load_layoutlm(self, entry: ModelEntry) -> dict:
        from transformers import AutoModel, AutoTokenizer
        local = str(Path(entry.local_path))
        tokenizer = AutoTokenizer.from_pretrained(local, local_files_only=True)
        model = AutoModel.from_pretrained(local, local_files_only=True)
        return {"model": model, "tokenizer": tokenizer, "framework": "transformers"}

    def _load_layoutlmv3(self, entry: ModelEntry) -> dict:
        from transformers import AutoModel, AutoProcessor
        local = str(Path(entry.local_path))
        processor = AutoProcessor.from_pretrained(local, local_files_only=True)
        model = AutoModel.from_pretrained(local, local_files_only=True)
        return {"model": model, "processor": processor, "framework": "transformers"}
