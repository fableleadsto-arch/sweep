"""Tests for the native neural stack (companion/neural/).

Covers the spec's core guarantees, all measured for real:

  * **Import hygiene** — importing ``companion`` and ``companion.neural`` never
    loads torch/safetensors/tokenizers (boot stays dependency-free).
  * **Config** — validation, serialization round-trip, scale presets.
  * **Architecture** — forward shapes, backward gives every param a gradient,
    GQA reduces KV memory, KV-cache decode == full recompute, RoPE determinism.
  * **Parameter counting** — analytical count matches the instantiated model.
  * **Tokenizers** — training, encode/decode round-trip, save/load, fallback.
  * **Training** — a real (tiny) training run reduces loss and checkpoints.
  * **Checkpoints** — safetensors weights round-trip exactly; tokenizer loads.
  * **Inference** — generation returns real tokens with measured tokens/sec.
  * **Evaluation** — the battery passes each gate and reports real numbers.
  * **Selection** — hardware detection + honest train-mode fit estimates.
  * **Registry & router** — records carry computed params; routing falls back
    to external when no trained model exists.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from companion.tools.common import module_available

HAS_TORCH = module_available("torch")
HAS_SAFETENSORS = module_available("safetensors")
HAS_TOKENIZERS = module_available("tokenizers")

needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
needs_tokenizers = pytest.mark.skipif(not HAS_TOKENIZERS, reason="tokenizers not installed")


def _tiny_config(**overrides) -> "object":
    from companion.neural.architecture import ModelConfig

    base = dict(
        name="relay-nano-test",
        vocab_size=512,
        hidden_size=64,
        intermediate_size=192,
        num_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_context_length=64,
    )
    base.update(overrides)
    return ModelConfig(**base)


def _tiny_corpus(seed: int = 0, n: int = 200) -> list[str]:
    rng = random.Random(seed)
    words = ["relay", "ai", "native", "neural", "network", "hello", "world", "the", "of", "and"]
    return [" ".join(rng.choices(words, k=rng.randint(4, 12))) for _ in range(n)]


# ── import hygiene ──────────────────────────────────────────────────────


def test_neural_package_import_never_loads_heavy_frameworks():
    import subprocess
    import sys

    code = (
        "import companion.neural\n"
        "import sys\n"
        "print([m for m in ('torch', 'safetensors', 'tokenizers') if m in sys.modules])\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=".")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


# ── config ──────────────────────────────────────────────────────────────


def test_model_config_validation():
    from companion.neural.architecture import ModelConfig

    with pytest.raises(ValueError):
        ModelConfig(vocab_size=1)
    with pytest.raises(ValueError):
        ModelConfig(hidden_size=10, num_attention_heads=4)
    with pytest.raises(ValueError):
        ModelConfig(normalization="bogus")
    with pytest.raises(ValueError):
        ModelConfig(num_attention_heads=4, num_key_value_heads=3)


def test_model_config_roundtrip_and_head_dim():
    from companion.neural.architecture import ModelConfig

    cfg = _tiny_config()
    cfg2 = ModelConfig.from_dict(cfg.to_dict())
    assert cfg2.to_dict() == cfg.to_dict()
    assert cfg.head_dim == cfg.hidden_size // cfg.num_attention_heads


def test_scale_presets_exist_and_are_valid():
    from companion.neural import SCALES
    from companion.neural.architecture import ModelConfig

    for name in ("nano", "small", "medium", "large", "x"):
        cfg = ModelConfig(**SCALES[name])
        assert cfg.name == f"relay-{name}"
    assert SCALES["x"]["status"] == "planned"


# ── architecture ────────────────────────────────────────────────────────


@needs_torch
def test_forward_backward_shapes_and_grads():
    import torch

    from companion.neural.architecture import RelayTransformer

    cfg = _tiny_config()
    model = RelayTransformer(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, hidden = model(ids)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert hidden.shape == (2, 16, cfg.hidden_size)

    model.train()
    tgt = torch.randint(0, cfg.vocab_size, (2, 16))
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, cfg.vocab_size), tgt[:, 1:].reshape(-1))
    loss.backward()
    total = sum(1 for p in model.parameters())
    with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert with_grad == total, f"expected grads on all {total} params, got {with_grad}"


@needs_torch
def test_kv_cache_decode_matches_full_recompute():
    import torch

    from companion.neural.architecture import RelayTransformer

    torch.manual_seed(0)
    cfg = _tiny_config()
    model = RelayTransformer(cfg)
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 12))
    full_logits, _ = model(ids)
    cache = model.new_cache()
    model(ids[:, :6], cache=cache)
    dec_logits, _ = model(ids[:, 6:], start_pos=6, cache=cache)
    assert len(cache) == 12
    assert torch.allclose(full_logits[:, 6:], dec_logits, atol=1e-5)


@needs_torch
def test_gqa_uses_less_kv_memory():
    import torch

    from companion.neural.architecture import RelayTransformer

    full = _tiny_config(num_key_value_heads=2)
    gqa = _tiny_config(num_key_value_heads=1)
    m_full = RelayTransformer(full)
    m_gqa = RelayTransformer(gqa)
    ids = torch.randint(0, full.vocab_size, (1, 8))
    m_full(ids)
    m_gqa(ids)
    kv_full = m_full.layers[0].self_attn.k_proj.weight.numel()
    kv_gqa = m_gqa.layers[0].self_attn.k_proj.weight.numel()
    assert kv_gqa < kv_full
    assert m_gqa.param_count() < m_full.param_count()


@needs_torch
def test_rope_deterministic_across_batches():
    import torch

    from companion.neural.architecture import RelayTransformer

    cfg = _tiny_config()
    m1 = RelayTransformer(cfg)
    m1.eval()
    m2 = RelayTransformer(cfg)
    m2.eval()
    m2.load_state_dict(m1.state_dict())
    ids = torch.tensor([[1, 2, 3, 4]])
    l1, _ = m1(ids)
    l2, _ = m2(ids)
    assert torch.equal(l1, l2)


@needs_torch
def test_parameter_count_matches_instantiation():
    from companion.neural.architecture import RelayTransformer
    from companion.neural.registry import _count_parameters_from_config

    for cfg in (_tiny_config(), _tiny_config(tie_word_embeddings=True)):
        model = RelayTransformer(cfg)
        assert _count_parameters_from_config(cfg) == model.param_count()


# ── tokenizer ───────────────────────────────────────────────────────────


@needs_tokenizers
def test_tokenizer_train_roundtrip_save_load(tmp_path):
    from companion.neural.tokenizer import train_tokenizer

    corpus = _tiny_corpus()
    tok = train_tokenizer(corpus, vocab_size=512)
    assert not tok.is_fallback
    sample = corpus[0]
    ids = tok.encode(sample)
    assert tok.decode(ids) == sample
    path = str(tmp_path / "tokenizer.json")
    tok.save(path)
    tok2 = train_tokenizer(["x"], vocab_size=16)
    tok2 = tok2.__class__.load(path)
    assert tok2.decode(tok2.encode(sample)) == sample


def test_tokenizer_manifest():
    from companion.neural.tokenizer import train_tokenizer

    tok = train_tokenizer(["hello world"], vocab_size=64)
    m = tok.to_manifest()
    assert m["type"] == "bpe" and m["vocab_size"] > 0


# ── training + checkpoints ──────────────────────────────────────────────


@needs_torch
@needs_tokenizers
def test_training_reduces_loss_and_checkpoints(tmp_path):
    import torch

    from companion.neural.architecture import RelayTransformer
    from companion.neural.tokenizer import train_tokenizer
    from companion.neural.training import TextDataset, TrainConfig, load_model, load_tokenizer, train

    cfg = _tiny_config()
    tok = train_tokenizer(_tiny_corpus(), vocab_size=cfg.vocab_size)
    tok.save(str(tmp_path / "tokenizer.json"))
    model = RelayTransformer(cfg)
    ds = TextDataset.from_texts(_tiny_corpus(), tok, source="test", tokenizer_path=str(tmp_path / "tokenizer.json"))
    ckpt = str(tmp_path / "ckpt")

    losses: list[float] = []
    tc = TrainConfig(batch_size=4, seq_len=16, total_steps=4, warmup_steps=2, log_every=1, device="cpu")
    train(model, tok, lambda: ds.to_dataloader(4, 16, seed=1), tc, checkpoint_dir=ckpt, on_log=lambda e: losses.append(e["loss"]))

    assert len(losses) == 4
    assert losses[-1] < losses[0], f"expected loss to decrease: {losses}"

    # checkpoint round-trip: weights identical after reload
    m2 = load_model(ckpt)
    s1, s2 = model.state_dict(), m2.state_dict()
    assert set(s1) == set(s2)
    assert all(torch.equal(s1[k], s2[k]) for k in s1)
    t2 = load_tokenizer(ckpt)
    assert t2.decode(t2.encode("relay ai")) == "relay ai"


@needs_torch
def test_checkpoint_without_tokenizer_roundtrips(tmp_path):
    import torch

    from companion.neural.architecture import RelayTransformer
    from companion.neural.training.checkpointing import load_model, save_checkpoint

    model = RelayTransformer(_tiny_config())
    ckpt = str(tmp_path / "ckpt")
    save_checkpoint(model, None, None, 0, float("inf"), ckpt)
    m2 = load_model(ckpt)
    s1, s2 = model.state_dict(), m2.state_dict()
    assert all(torch.equal(s1[k], s2[k]) for k in s1)


# ── inference ───────────────────────────────────────────────────────────


@needs_torch
@needs_tokenizers
def test_generation_returns_real_tokens(tmp_path):
    from companion.neural.architecture import RelayTransformer
    from companion.neural.inference import GenerationConfig, generate
    from companion.neural.tokenizer import train_tokenizer
    from companion.neural.training import TextDataset, TrainConfig, load_model, load_tokenizer, train

    cfg = _tiny_config()
    tok = train_tokenizer(_tiny_corpus(), vocab_size=cfg.vocab_size)
    tok.save(str(tmp_path / "tokenizer.json"))
    model = RelayTransformer(cfg)
    ds = TextDataset.from_texts(_tiny_corpus(), tok, source="test", tokenizer_path=str(tmp_path / "tokenizer.json"))
    tc = TrainConfig(batch_size=4, seq_len=16, total_steps=4, warmup_steps=2, device="cpu")
    train(model, tok, lambda: ds.to_dataloader(4, 16, seed=1), tc, checkpoint_dir=str(tmp_path / "ckpt"))
    m = load_model(str(tmp_path / "ckpt"))
    t = load_tokenizer(str(tmp_path / "ckpt"))

    gen = generate(m, t, "relay", GenerationConfig(max_new_tokens=8, temperature=0.0, seed=1), device="cpu")
    assert gen.generated_tokens > 0
    assert gen.tokens_per_second > 0
    assert len(gen.text) > 0


@needs_torch
def test_sampling_respects_top_k():
    import torch

    from companion.neural.inference import sample_token

    logits = torch.randn(100)
    rng = torch.Generator().manual_seed(0)
    ids = [sample_token(logits, top_k=1, rng=rng) for _ in range(5)]
    assert all(i == int(logits.argmax()) for i in ids)


# ── evaluation ──────────────────────────────────────────────────────────


@needs_torch
@needs_tokenizers
def test_evaluation_battery_passes(tmp_path):
    from companion.neural.architecture import RelayTransformer
    from companion.neural.evaluation import run_evaluation
    from companion.neural.tokenizer import train_tokenizer

    cfg = _tiny_config()
    tok = train_tokenizer(_tiny_corpus(), vocab_size=cfg.vocab_size)
    tok.save(str(tmp_path / "tokenizer.json"))
    model = RelayTransformer(cfg)
    results = run_evaluation(model, tok, device="cpu", checkpoint_dir=str(tmp_path / "ckpt"), tmp_dir=tmp_path)
    assert all(r["passed"] for r in results.values()), results


# ── selection ───────────────────────────────────────────────────────────


def test_detect_hardware_is_honest():
    from companion.neural.selection import detect_hardware

    hw = detect_hardware()
    assert hw["cpus"] > 0
    assert hw["ram_bytes"] > 0
    assert "fp32" in hw["precision_supported"]


@needs_torch
def test_fit_estimate_train_mode_is_conservative():
    from companion.neural.architecture import ModelConfig
    from companion.neural.selection import estimate_fit

    hw = {"ram_bytes": 512 * 1024**3, "cpus": 8, "gpu": False, "precision_supported": ["fp32", "bf16"]}
    cfg = ModelConfig(**dict(name="tiny", vocab_size=1000, hidden_size=64, intermediate_size=192, num_layers=1, num_attention_heads=2, num_key_value_heads=1, max_context_length=128))
    fit_train = estimate_fit(cfg, hw, mode="train")
    fit_infer = estimate_fit(cfg, hw, mode="infer")
    assert fit_train.footprint_bytes > fit_infer.footprint_bytes
    assert fit_infer.fits and fit_train.fits


# ── registry + router ───────────────────────────────────────────────────


def test_registry_records_computed_params(tmp_path):
    import torch

    from companion.neural.architecture import RelayTransformer
    from companion.neural.registry import ModelRegistry
    from companion.neural.training.checkpointing import save_checkpoint

    cfg = _tiny_config()
    model = RelayTransformer(cfg)
    save_checkpoint(model, None, None, 0, 1.0, str(tmp_path / "models" / "m1"))
    reg = ModelRegistry(tmp_path / "models")
    rec = reg.resolve(None)
    assert rec.parameters == model.param_count()
    assert rec.verified is True


def test_router_falls_back_without_trained_model(tmp_path):
    from companion.neural.registry import ModelRegistry
    from companion.neural.router import NativeRouter

    cfg = _tiny_config()
    from companion.neural.architecture import ModelConfig
    from companion.neural.training.checkpointing import save_checkpoint
    import torch as _t

    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    from companion.neural.architecture import RelayTransformer

    save_checkpoint(RelayTransformer(cfg), None, None, 0, float("inf"), str(model_dir / "m1"))
    reg = ModelRegistry(model_dir)
    router = NativeRouter(reg)
    decision = router.route("generation")
    assert decision.source == "native"
    assert decision.model is not None

    # A registry with an untrained model (no weights file) falls back.
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    (empty_dir / "config.json").write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
    reg2 = ModelRegistry(empty_dir)
    d2 = NativeRouter(reg2).route("generation")
    assert d2.source == "external"
    assert "no trained weights" in d2.reason


def test_router_unknown_task_falls_back(tmp_path):
    from companion.neural.registry import ModelRegistry
    from companion.neural.router import NativeRouter

    reg = ModelRegistry(tmp_path / "does_not_exist")
    d = NativeRouter(reg).route("make_art")
    assert d.source == "external"


# ── API endpoints ───────────────────────────────────────────────────────


def test_native_endpoints_respond():
    from fastapi.testclient import TestClient

    from companion.main import app

    with TestClient(app, headers={"Authorization": "Bearer test-only"}) as client:
        r = client.get("/api/brain/native/recommend")
        assert r.status_code == 200
        body = r.json()
        assert body["scale"] in {"relay-nano", "relay-small", "relay-medium", "relay-large", "relay-x"}

        r2 = client.post("/api/brain/native/route", json={"task": "generation"})
        assert r2.status_code == 200
        assert r2.json()["source"] in {"native", "external"}

        r3 = client.get("/api/brain/native/models")
        assert r3.status_code == 200
        assert isinstance(r3.json(), list)


@needs_torch
@needs_tokenizers
def test_native_generate_endpoint_end_to_end(tmp_path, monkeypatch):
    import companion.neural.registry as registry_mod
    from companion.main import app

    # Build a genuinely trained (tiny) model in the registry dir.
    from companion.neural.architecture import RelayTransformer
    from companion.neural.tokenizer import train_tokenizer
    from companion.neural.training import TextDataset, TrainConfig, train

    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True)
    monkeypatch.setattr(registry_mod, "DEFAULT_REGISTRY_DIR", model_dir)

    cfg = _tiny_config()
    tok = train_tokenizer(_tiny_corpus(), vocab_size=cfg.vocab_size)
    tok.save(str(model_dir / "tokenizer.json"))
    model = RelayTransformer(cfg)
    ds = TextDataset.from_texts(_tiny_corpus(), tok, source="test", tokenizer_path=str(model_dir / "tokenizer.json"))
    tc = TrainConfig(batch_size=4, seq_len=16, total_steps=4, warmup_steps=2, device="cpu")
    train(model, tok, lambda: ds.to_dataloader(4, 16, seed=1), tc, checkpoint_dir=str(model_dir / "relay-nano-test"))

    from fastapi.testclient import TestClient

    with TestClient(app, headers={"Authorization": "Bearer test-only"}) as client:
        r = client.get("/api/brain/native/models")
        assert r.status_code == 200
        records = r.json()
        assert len(records) == 1
        assert records[0]["parameters"] == model.param_count()
        assert records[0]["verified"] is True

        g = client.post("/api/brain/native/generate", json={"prompt": "relay", "max_new_tokens": 6, "temperature": 0.0})
        assert g.status_code == 200
        body = g.json()
        assert body["ok"] is True
        assert body["generated_tokens"] > 0
        assert body["model"] == "relay-nano-test"
        assert body["source"] == "native"
