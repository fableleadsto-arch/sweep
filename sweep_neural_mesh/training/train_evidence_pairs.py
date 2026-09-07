"""
Pair-format evidence classifier retraining.

Fixes the verified train/inference mismatch (SWEEP_ARCHITECTURE_AUDIT.md A1):
the deployed evidence classifier was fine-tuned on SINGLE sentences
(retrain_all.py: tokenizer(texts)) but is served at inference with
(premise, hypothesis) PAIRS (neural_engine.classify_evidence ->
tokenizer(premise, hypothesis), called from cortex._try_neural_fast_path).
Every inference input was out-of-distribution and the model collapsed to a
single label — 43.1% of all measured benchmark failures.

This script:
  1. Backs up the current artifact to evidence_classifier_v1_backup/
  2. Generates balanced (premise, hypothesis, label) pairs whose ENCODING
     matches inference exactly: tokenizer(premise, hypothesis)
  3. Fine-tunes bert-base-uncased on CPU within a ~1 GB memory budget
     (encoder frozen except the top 2 layers)
  4. Writes metadata.json with input_format="pair" — asserted by
     tests/test_neural_contract.py

Subjects/predicates are deliberately DISJOINT from the benchmark suite's
synthetic vocabulary (no zorp/quark/gloop/... here): the model must learn
negation/hedging/entailment STRUCTURE, not benchmark words (fairness rule,
directive §4).

Usage:
    python -m sweep_neural_mesh.training.train_evidence_pairs
"""
from __future__ import annotations

import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

OUTPUT_BASE = Path(__file__).parent / "neural_models"
ARTIFACT_DIR = OUTPUT_BASE / "evidence_classifier"
BACKUP_DIR = OUTPUT_BASE / "evidence_classifier_v1_backup"
BASE_MODEL = "bert-base-uncased"
SEED = 13

# ────────────────────────────────────────────────────────────────────
# 1. Vocabulary — disjoint from the benchmark suite's synthetic words
# ────────────────────────────────────────────────────────────────────

# Invented subjects (NOT used by the benchmark suite) + ordinary English nouns
SUBJECTS = [
    "krin", "vost", "pelar", "tremn", "durn", "slavik", "ormet", "fenwick",
    "sensor", "pump", "server", "relay", "valve", "module", "node", "link",
    "agent", "engine", "meter", "router",
]

# Status predicates (ordinary English status words; no benchmark-specific terms)
PREDICATES = [
    "stable", "enabled", "visible", "reachable", "calibrated", "responsive",
    "registered", "balanced", "reserved", "activated", "synchronized",
    "monitored", "isolated", "encrypted",
]

PREDICATE2 = ["offline", "dormant", "idle", "unresponsive", "corrupted"]

REF_LOGS = [f"diagnostic log {n}" for n in (2, 3, 5, 8, 12, 21)]
SOURCE_TAGS = ["Source A", "Source B", "the field team", "the control room"]


def _tpl_support(s: str, p: str) -> list[str]:
    return [
        f"The {s} is {p} and running smoothly.",
        f"Diagnostics confirm the {s} is {p}.",
        f"The log shows the {s} is {p}.",
        f"Operators report that the {s} is {p}.",
        f"The {s} is currently {p}.",
        f"Per the latest report, the {s} is {p}.",
        f"Status check: the {s} is {p}, verified twice.",
        f"{random.choice(SOURCE_TAGS)} says the {s} is {p}.",
    ]


def _tpl_refute(s: str, p: str) -> list[str]:
    log = random.choice(REF_LOGS)
    return [
        f"The {s} is not {p}.",
        f"It is false that the {s} is {p}.",
        f"The {s} is not {p} per {log}.",
        f"The log shows the {s} is not {p}.",
        f"Contrary to the rumor, the {s} is not {p}.",
        f"The {s} is currently not {p}.",
        f"{random.choice(SOURCE_TAGS)} says the {s} is not {p}.",
        f"The {s} is not {p}, according to the maintenance record.",
    ]


def _tpl_irrelevant(s: str, p: str) -> list[str]:
    """Evidence about a DIFFERENT subject/predicate — does not bear on the claim."""
    return [
        f"The {s} is {p}.",
        f"The {s} is not {p}.",
        f"{random.choice(SOURCE_TAGS)} mentions the {s} is {p}.",
        f"The log shows the {s} is {p}.",
    ]


def _tpl_hedged(s: str, p: str) -> list[str]:
    """Hedged evidence — does not verify the claim (correct label: neutral)."""
    return [
        f"The {s} could be {p}.",
        f"The {s} might be {p}.",
        f"The {s} may be {p}, but this is unconfirmed.",
        f"Someone claimed the {s} is {p}, though it is unverified.",
        f"The {s} will probably be {p} soon.",
    ]


def generate_pairs(rng: random.Random) -> list[tuple[str, str, int]]:
    """Return (premise, hypothesis, label) triples.

    Labels: 0=supports, 1=refutes, 2=neutral — matching
    neural_engine._ModelLoader._ev_labels.
    """
    pairs: list[tuple[str, str, int]] = []

    def pick() -> tuple[str, str]:
        return rng.choice(SUBJECTS), rng.choice(PREDICATES)

    # supports: positive claim verified by the evidence
    for _ in range(340):
        s, p = pick()
        premise = rng.choice(_tpl_support(s, p))
        pairs.append((premise, f"The {s} is {p}.", 0))
    # supports: negative claim verified by negative evidence
    for _ in range(60):
        s, p = pick()
        premise = rng.choice(_tpl_refute(s, p))
        pairs.append((premise, f"The {s} is not {p}.", 0))

    # refutes: positive claim contradicted by negative evidence
    for _ in range(340):
        s, p = pick()
        premise = rng.choice(_tpl_refute(s, p))
        pairs.append((premise, f"The {s} is {p}.", 1))
    # refutes: negative claim contradicted by positive evidence
    for _ in range(60):
        s, p = pick()
        premise = rng.choice(_tpl_support(s, p))
        pairs.append((premise, f"The {s} is not {p}.", 1))

    # neutral: evidence about a different subject/predicate
    for _ in range(220):
        s, p = pick()
        s2 = rng.choice([x for x in SUBJECTS if x != s])
        p2 = rng.choice(PREDICATES + PREDICATE2)
        if rng.random() < 0.5:
            s2, p2, s, p = s, p, s2, p2  # sometimes claim has the novel subject
        premise = rng.choice(_tpl_irrelevant(s2, p2))
        pairs.append((premise, f"The {s} is {p}.", 2))

    # neutral: hedged evidence about the same subject — not verification
    for _ in range(180):
        s, p = pick()
        premise = rng.choice(_tpl_hedged(s, p))
        pairs.append((premise, f"The {s} is {p}.", 2))

    # dedupe + deterministic shuffle
    pairs = sorted(set(pairs))
    rng.shuffle(pairs)
    return pairs


# ────────────────────────────────────────────────────────────────────
# 2. Training — CPU, memory-budgeted (~1 GB)
# ────────────────────────────────────────────────────────────────────

def train(pairs, output_dir: str, epochs: int = 6, batch_size: int = 8,
          lr: float = 2e-5) -> float:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    torch.manual_seed(SEED)
    torch.set_num_threads(min(8, os.cpu_count() or 4))

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=3
    )

    # Memory budget: freeze everything except the top 2 encoder layers,
    # pooler and classifier head. Only trainable params hold grads + AdamW state.
    n_trainable = 0
    for name, param in model.named_parameters():
        trainable = name.startswith("bert.encoder.layer.9") or \
            name.startswith("bert.encoder.layer.10") or \
            name.startswith("bert.encoder.layer.11") or \
            name.startswith("bert.pooler") or \
            name.startswith("classifier")
        param.requires_grad = trainable
        n_trainable += param.numel() if trainable else 0
    print(f"Trainable parameters: {n_trainable / 1e6:.1f}M "
          f"({n_trainable / sum(p.numel() for p in model.parameters()):.1%} of model)")

    premises = [p for p, _, _ in pairs]
    hypotheses = [h for _, h, _ in pairs]
    labels = [l for _, _, l in pairs]

    # Pair encoding — EXACTLY what inference does:
    #   tokenizer(premise, hypothesis) -> [CLS] premise [SEP] hypothesis [SEP]
    enc = tokenizer(premises, hypotheses, truncation=True, padding=True,
                    max_length=64, return_tensors="pt")
    labels_t = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(enc["input_ids"], enc["attention_mask"], labels_t)

    split = int(len(dataset) * 0.9)
    train_ds = TensorDataset(enc["input_ids"][:split], enc["attention_mask"][:split],
                             labels_t[:split])
    val_ds = TensorDataset(enc["input_ids"][split:], enc["attention_mask"][split:],
                           labels_t[split:])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = correct = total = 0
        for input_ids, attention_mask, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.logits.argmax(dim=1) == batch_labels).sum().item()
            total += len(batch_labels)
        scheduler.step()
        train_acc = correct / total

        model.eval()
        val_correct = val_total = 0
        per_label = {0: [0, 0], 1: [0, 0], 2: [0, 0]}
        with torch.no_grad():
            for input_ids, attention_mask, batch_labels in val_loader:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.argmax(dim=1)
                val_correct += (preds == batch_labels).sum().item()
                val_total += len(batch_labels)
                for lbl, pred in zip(batch_labels.tolist(), preds.tolist()):
                    per_label[lbl][1] += 1
                    per_label[lbl][0] += int(lbl == pred)
        val_acc = val_correct / val_total if val_total else 0.0

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

        detail = " ".join(
            f"L{k}:{v[0]}/{v[1]}" for k, v in per_label.items() if v[1]
        )
        print(f"  epoch {epoch + 1}/{epochs}: loss={total_loss / max(len(train_loader), 1):.4f} "
              f"train_acc={train_acc:.1%} val_acc={val_acc:.1%} ({detail}) "
              f"{'*SAVED*' if val_acc == best_val_acc else ''}", flush=True)

    return best_val_acc


def main() -> None:
    t0 = time.perf_counter()
    rng = random.Random(SEED)
    pairs = generate_pairs(rng)
    from collections import Counter
    dist = Counter(l for _, _, l in pairs)
    print(f"Generated {len(pairs)} unique pairs; class distribution: "
          f"supports={dist[0]} refutes={dist[1]} neutral={dist[2]}")

    # 1. Backup current (single-sentence-trained) artifact
    if ARTIFACT_DIR.exists() and not BACKUP_DIR.exists():
        shutil.copytree(ARTIFACT_DIR, BACKUP_DIR)
        print(f"Backed up old artifact -> {BACKUP_DIR.name}/")

    # 2. Train
    print(f"Training pair-format evidence classifier on CPU ({BASE_MODEL})...")
    best = train(pairs, str(ARTIFACT_DIR))

    # 3. Metadata — input_format is asserted by tests/test_neural_contract.py
    metadata = {
        "model": f"{BASE_MODEL}-pair-finetuned",
        "task": "evidence_classification",
        "input_format": "pair",
        "encoding": "tokenizer(premise, hypothesis) — matches neural_engine.classify_evidence",
        "labels": ["supports", "refutes", "neutral"],
        "training_examples": len(pairs),
        "class_distribution": {"supports": dist[0], "refutes": dist[1], "neutral": dist[2]},
        "epochs": 6,
        "frozen": "embeddings + encoder layers 0-8",
        "best_val_accuracy": round(best, 4),
        "seed": SEED,
        "trained_by": "sweep_neural_mesh.training.train_evidence_pairs",
        "status": "fine-tuned",
        "replaces": "retrain_all.py single-sentence artifact (degenerate; see SWEEP_ARCHITECTURE_AUDIT.md A1)",
    }
    (ARTIFACT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.0f}s. Best val accuracy: {best:.1%}")
    print(f"Artifact: {ARTIFACT_DIR}")

    # 4. Quick self-check through the production inference path
    try:
        from sweep_neural_mesh.neurons.neural_engine import NeuralEngine
        eng = NeuralEngine()
        if eng.wait_until_ready(timeout=180):
            probes = [
                ("The krin is stable and running smoothly.", "The krin is stable.", "supports"),
                ("The vost is not enabled per diagnostic log 3.", "The vost is enabled.", "refutes"),
                ("The tremn is awake.", "The solk is visible.", "neutral"),
                ("The durn could be locked.", "The durn is locked.", "neutral"),
            ]
            ok = 0
            for premise, hypothesis, expected in probes:
                r = eng.classify_evidence(premise, hypothesis)
                good = r.ready and r.label == expected and r.confidence > 0.5
                ok += bool(good)
                print(f"  [{'OK' if good else 'MISS'}] {r.label}@{r.confidence:.2f} "
                      f"(expected {expected}) | {premise[:40]}... <-> {hypothesis}")
            print(f"Self-check through production path: {ok}/{len(probes)}")
    except Exception as e:  # pragma: no cover
        print(f"Self-check skipped: {e}")


if __name__ == "__main__":
    main()
