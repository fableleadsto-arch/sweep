"""End-to-end tests for the Sweep Intelligence Intent Core pipeline.

Tests real model inference with downloaded checkpoints — no mocks, no fakes.
"""

import os
import sys
import time
import tempfile
from pathlib import Path

import pytest
import numpy as np

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# Auto-detect Tesseract on Windows
if sys.platform == "win32" and not os.environ.get("PATH", "").count("Tesseract"):
    tess = r"C:\Program Files\Tesseract-OCR"
    if os.path.isdir(tess):
        os.environ["PATH"] = tess + ";" + os.environ.get("PATH", "")

from services.intelligence.model_manager.device import detect_device
from services.intelligence.model_manager.registry import ModelRegistry, ModelStatus
from services.intelligence.model_manager.loader import ModelLoader
from services.intelligence.intent_core import IntentCore, Modality


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def device():
    return detect_device()


@pytest.fixture(scope="module")
def registry(tmp_path_factory):
    r = ModelRegistry(models_root=str(tmp_path_factory.mktemp("model-status")))
    # Force status to downloaded for all models that have weights on disk
    r.sync_from_disk()
    return r


@pytest.fixture(scope="module")
def loader(registry, device):
    return ModelLoader(registry, device)


@pytest.fixture(scope="module")
def intent_core(device, registry):
    core = IntentCore(device)
    core.registry = registry
    core.loader = ModelLoader(registry, device)
    return core


def _skip_if_unavailable(name, registry):
    entry = registry.get(name)
    if not entry or entry.status == ModelStatus.NOT_DOWNLOADED:
        pytest.skip(f"{name} not downloaded")


# ── Device Tests ──────────────────────────────────────────────────────────

class TestDevice:
    def test_detection(self, device):
        assert device.device in ("cpu", "cuda", "mps")
        assert device.ram_total_gb > 0
        assert device.cpu_count > 0

    def test_cuda_optional(self, device):
        # CUDA is optional — CPU-only is fine
        if device.cuda_available:
            assert device.gpu_name != ""
            assert device.gpu_memory_gb > 0


# ── Registry Tests ────────────────────────────────────────────────────────

class TestRegistry:
    def test_count(self, registry):
        assert len(registry.all()) >= 18

    def test_categories(self, registry):
        for cat in ["nlp", "embeddings", "vision", "detection", "face", "audio", "speech", "documents"]:
            entries = registry.by_category(cat)
            assert len(entries) > 0, f"No models in category: {cat}"

    def test_lookup(self, registry):
        entry = registry.get("bert-base-uncased")
        assert entry is not None
        assert entry.checkpoint == "bert-base-uncased"

    def test_capabilities_map(self, registry):
        for name in ["gliner-base-v2.1", "deberta-v3-base", "all-MiniLM-L6-v2", "clip-vit-b32"]:
            entry = registry.get(name)
            assert entry is not None, f"Model {name} not in registry"

    def test_installed_count(self, registry):
        installed = registry.installed()
        assert len(installed) >= 13, f"Expected >=13 installed, got {len(installed)}"


# ── NLP Model Tests ──────────────────────────────────────────────────────

class TestNLPModels:
    def test_bert_inference(self, loader, registry):
        _skip_if_unavailable("bert-base-uncased", registry)
        loaded = loader.load("bert-base-uncased")
        import torch
        tokenizer = loaded["tokenizer"]
        model = loaded["model"]
        inputs = tokenizer("Hello world", return_tensors="pt", truncation=True)
        with torch.no_grad():
            out = model(**inputs)
        emb = out.last_hidden_state.mean(dim=1)
        assert emb.shape == (1, 768), f"Expected (1, 768), got {emb.shape}"
        loader.unload("bert-base-uncased")

    def test_deberta_inference(self, loader, registry):
        _skip_if_unavailable("deberta-v3-base", registry)
        loaded = loader.load("deberta-v3-base")
        import torch
        tokenizer = loaded["tokenizer"]
        model = loaded["model"]
        inputs = tokenizer("Intent classification test", return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            out = model(**inputs)
        assert out.last_hidden_state.shape[2] == 768
        loader.unload("deberta-v3-base")

    def test_gliner_entity_extraction(self, loader, registry):
        _skip_if_unavailable("gliner-base-v2.1", registry)
        loaded = loader.load("gliner-base-v2.1")
        model = loaded["model"]
        text = "Find information about John Smith associated with Acme Corp in Delhi during 2023."
        labels = ["PERSON", "ORGANIZATION", "LOCATION", "DATE"]
        entities = model.predict_entities(text, labels, threshold=0.3)
        entity_texts = [e["text"] for e in entities]
        # Should find at least some entities
        assert len(entities) >= 2, f"Expected >=2 entities, got {entities}"
        assert "PERSON" in [e["label"] for e in entities], f"Expected PERSON in labels: {entities}"
        assert "John Smith" in entity_texts or any("John" in t for t in entity_texts)
        loader.unload("gliner-base-v2.1")

    def test_xlm_multilingual(self, loader, registry):
        _skip_if_unavailable("xlm-roberta-base", registry)
        loaded = loader.load("xlm-roberta-base")
        import torch
        tokenizer = loaded["tokenizer"]
        model = loaded["model"]
        # Test with non-English text
        texts = ["Bonjour le monde", "Hola mundo", "Привет мир"]
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                out = model(**inputs)
            assert out.last_hidden_state.shape[2] == 768
        loader.unload("xlm-roberta-base")


# ── Embedding Model Tests ─────────────────────────────────────────────────

class TestEmbeddingModels:
    def test_minilm_encoding(self, loader, registry):
        _skip_if_unavailable("all-MiniLM-L6-v2", registry)
        loaded = loader.load("all-MiniLM-L6-v2")
        model = loaded["model"]
        sentences = [
            "Find information about John Smith",
            "Who is John Smith?",
            "The weather is nice today",
        ]
        embeddings = model.encode(sentences, normalize_embeddings=True)
        assert embeddings.shape == (3, 384), f"Expected (3, 384), got {embeddings.shape}"
        # Similar sentences should be more similar than unrelated
        sim_related = np.dot(embeddings[0], embeddings[1])
        sim_unrelated = np.dot(embeddings[0], embeddings[2])
        assert sim_related > sim_unrelated, f"Related ({sim_related:.3f}) should be > unrelated ({sim_unrelated:.3f})"
        loader.unload("all-MiniLM-L6-v2")

    def test_mpnet_encoding(self, loader, registry):
        _skip_if_unavailable("all-mpnet-base-v2", registry)
        loaded = loader.load("all-mpnet-base-v2")
        model = loaded["model"]
        embeddings = model.encode(["Hello world", "Test sentence"])
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] == 768  # MPNet is 768-dim
        loader.unload("all-mpnet-base-v2")


# ── Vision Model Tests ────────────────────────────────────────────────────

class TestVisionModels:
    def test_clip_text_encoding(self, loader, registry):
        _skip_if_unavailable("clip-vit-b32", registry)
        loaded = loader.load("clip-vit-b32")
        import torch
        tokenizer = loaded["tokenizer"]
        model = loaded["model"]
        tokens = tokenizer(["a photo of a cat", "a photo of a dog"])
        with torch.no_grad():
            features = model.encode_text(tokens)
        assert features.shape == (2, 512), f"Expected (2, 512), got {features.shape}"
        loader.unload("clip-vit-b32")

    def test_yolo_detection(self, loader, registry):
        _skip_if_unavailable("yolo-v8n", registry)
        loaded = loader.load("yolo-v8n")
        model = loaded["model"]
        # Create a test image
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        results = model(img, verbose=False)
        assert len(results) == 1
        assert hasattr(results[0], "boxes")
        loader.unload("yolo-v8n")

    def test_opencv_processing(self):
        """Test OpenCV engine — no model download needed."""
        from services.intelligence.vision.opencv_engine import OpenCVEngine
        engine = OpenCVEngine()
        assert engine.available()
        # Create and process test image
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        gray = engine.to_grayscale(img)
        assert gray.shape == (100, 100)
        edges = engine.detect_edges(img)
        assert edges.shape == (100, 100)
        resized = engine.resize(img, max_dim=50)
        assert resized.shape[0] <= 50 and resized.shape[1] <= 50


# ── Face Model Tests ──────────────────────────────────────────────────────

class TestFaceModels:
    def test_insightface_detection(self, loader, registry):
        _skip_if_unavailable("insightface", registry)
        loaded = loader.load("insightface")
        app = loaded["model"]
        # Blank image — should return 0 faces (not an error)
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        faces = app.get(img)
        assert isinstance(faces, list)
        # Blank image should have 0 faces
        assert len(faces) == 0
        loader.unload("insightface")


# ── Audio Model Tests ─────────────────────────────────────────────────────

class TestAudioModels:
    def test_whisper_transcription(self, loader, registry):
        _skip_if_unavailable("whisper-base", registry)
        loaded = loader.load("whisper-base")
        model = loaded["model"]
        # Silent audio
        audio = np.zeros(16000, dtype=np.float32)
        result = model.transcribe(audio, fp16=False)
        assert "text" in result
        assert "segments" in result
        assert isinstance(result["segments"], list)
        loader.unload("whisper-base")

    def test_wav2vec2_encoding(self, loader, registry):
        _skip_if_unavailable("wav2vec2-base", registry)
        loaded = loader.load("wav2vec2-base")
        import torch
        model = loaded["model"]
        processor = loaded["processor"]
        audio = np.zeros(16000, dtype=np.float32)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        assert outputs.last_hidden_state.shape[2] == 768
        loader.unload("wav2vec2-base")


# ── Document Model Tests ──────────────────────────────────────────────────

class TestDocumentModels:
    def test_layoutlm_inference(self, loader, registry):
        _skip_if_unavailable("layoutlm-base", registry)
        loaded = loader.load("layoutlm-base")
        import torch
        tokenizer = loaded["tokenizer"]
        model = loaded["model"]
        words = ["Invoice", "from", "Acme", "Corp"]
        n = len(words)
        boxes = [[0, 0, 0, 0]] + [[int(i * 1000 / n), 10, int((i + 1) * 1000 / n), 90] for i in range(n)] + [[0, 0, 0, 0]]
        enc = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=32)
        seq_len = enc["input_ids"].shape[1]
        while len(boxes) < seq_len:
            boxes.append([0, 0, 0, 0])
        enc["bbox"] = torch.tensor([boxes[:seq_len]])
        with torch.no_grad():
            out = model(**enc)
        assert out.last_hidden_state.shape[2] == 768
        loader.unload("layoutlm-base")

    def test_tesseract_ocr(self):
        """Test Tesseract OCR."""
        from services.intelligence.document.ocr_engine import OCREngine
        engine = OCREngine()
        if not engine.available():
            pytest.skip("Tesseract not installed")
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (400, 100), "white")
        d = ImageDraw.Draw(img)
        d.text((10, 30), "Hello OCR World 2024", fill="black")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
            img.save(path)
        text = engine.ocr_image(path)
        os.unlink(path)
        assert "Hello" in text
        assert "OCR" in text or "ocr" in text.lower()


# ── Intent Core Integration Tests ─────────────────────────────────────────

class TestIntentCore:
    def test_modality_detection_text(self, intent_core):
        modality = intent_core.detect_modality("Hello world")
        assert modality == Modality.TEXT

    def test_modality_detection_image(self, intent_core):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
            path = f.name
        modality = intent_core.detect_modality(path)
        os.unlink(path)
        assert modality == Modality.IMAGE

    def test_modality_detection_audio(self, intent_core):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF" + b"\x00" * 100)
            path = f.name
        modality = intent_core.detect_modality(path)
        os.unlink(path)
        assert modality == Modality.AUDIO

    def test_text_processing(self, intent_core, registry):
        """Full text pipeline: entity extraction + embedding."""
        text = "Find information about John Smith associated with Acme in Delhi during 2023."
        intent_core.reset()
        context = intent_core.process(text)
        # Should have at least one observation
        assert len(context.observations) >= 1
        # Entity extraction via GLiNER
        _skip_if_unavailable("gliner-base-v2.1", registry)
        entities = intent_core.extract_entities(text)
        assert len(entities) >= 2
        entity_labels = [e["label"] for e in entities]
        assert "PERSON" in entity_labels

    def test_entity_extraction(self, intent_core, registry):
        """Direct entity extraction test."""
        _skip_if_unavailable("gliner-base-v2.1", registry)
        text = "John Smith works at Acme Corp in New York since 2023."
        entities = intent_core.extract_entities(text)
        assert len(entities) >= 2
        texts = [e["text"] for e in entities]
        labels = [e["label"] for e in entities]
        assert any("John" in t or "Smith" in t for t in texts), f"Expected PERSON: {entities}"
        assert "PERSON" in labels

    def test_text_embedding(self, intent_core, registry):
        """Embedding generation and similarity."""
        _skip_if_unavailable("all-MiniLM-L6-v2", registry)
        emb1 = intent_core.embed_text("Find John Smith")
        emb2 = intent_core.embed_text("Who is John Smith?")
        emb3 = intent_core.embed_text("The stock market crashed")
        assert len(emb1) == 384
        assert len(emb2) == 384
        # Related queries should be more similar
        sim_related = sum(a * b for a, b in zip(emb1, emb2))
        sim_unrelated = sum(a * b for a, b in zip(emb1, emb3))
        assert sim_related > sim_unrelated

    def test_vector_search(self, intent_core, registry):
        """FAISS nearest-neighbor search."""
        _skip_if_unavailable("all-MiniLM-L6-v2", registry)
        query = intent_core.embed_text("John Smith at Acme")
        candidates = [
            intent_core.embed_text("John Smith works at Acme Corp"),
            intent_core.embed_text("Alice Johnson at Beta Inc"),
            intent_core.embed_text("John Smith's address in New York"),
        ]
        results = intent_core.search_similar(query, candidates, top_k=2)
        assert len(results) == 2
        assert results[0]["index"] == 0  # most similar should be first
        # Clean up observations
        intent_core.reset()

    def test_context_accumulation(self, intent_core, registry):
        """Verify observations accumulate in context."""
        _skip_if_unavailable("gliner-base-v2.1", registry)
        intent_core.reset()
        intent_core.extract_entities("John Smith at Acme in Delhi 2023")
        intent_core.extract_entities("Alice Johnson at Beta in London 2024")
        assert len(intent_core.context.observations) >= 2
        assert len(intent_core.context.entities) >= 4


# ── Cache Tests ───────────────────────────────────────────────────────────

class TestCache:
    def test_put_get(self):
        from services.intelligence.model_manager.cache import ModelCache
        cache = ModelCache(max_size=3, max_memory_mb=100)
        cache.put("m1", "model1", size_mb=10)
        assert cache.get("m1") == "model1"
        assert cache.count == 1

    def test_eviction(self):
        from services.intelligence.model_manager.cache import ModelCache
        cache = ModelCache(max_size=2, max_memory_mb=100)
        cache.put("m1", "model1", size_mb=10)
        cache.put("m2", "model2", size_mb=10)
        cache.put("m3", "model3", size_mb=10)  # should evict m1
        assert cache.count == 2
        assert cache.get("m1") is None
        assert cache.get("m3") == "model3"

    def test_stats(self):
        from services.intelligence.model_manager.cache import ModelCache
        cache = ModelCache(max_size=4, max_memory_mb=500)
        cache.put("m1", "x", size_mb=100)
        stats = cache.stats()
        assert stats["count"] == 1
        assert stats["total_mb"] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
