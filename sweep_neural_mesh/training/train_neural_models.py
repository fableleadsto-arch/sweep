"""
Train Neural Models — Fine-tune BERT classifiers on the comprehensive dataset.

Trains three models:
1. Evidence Classifier (supports/refutes/neutral)
2. Contradiction Detector (contradiction/consistent/partial)
3. Intent Classifier (investigation/search/comparison/etc.)

Uses the existing comprehensive_train.jsonl dataset.
Saves fine-tuned models to neural_models/ directory.

CPU-only compatible. Uses deterministic seeds.

Usage:
    python -m sweep_neural_mesh.training.train_neural_models
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

# Suppress TF warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_neural_models")

# Ensure correct paths
_sweep_dir = Path(__file__).resolve().parent.parent
_sweep_parent = _sweep_dir.parent
sys.path.insert(0, str(_sweep_parent))
sys.path.insert(0, str(_sweep_dir))

# Deterministic seeds
SEED = 42
random.seed(SEED)


def load_dataset(split: str = "train") -> list[dict]:
    """Load a dataset split."""
    path = _sweep_dir / "training" / "datasets" / f"comprehensive_{split}.jsonl"
    if not path.exists():
        logger.error(f"Dataset not found: {path}")
        return []
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def train_evidence_classifier(train_data: list[dict], val_data: list[dict]) -> bool:
    """Fine-tune BERT for evidence classification."""
    try:
        import torch
        from torch.utils.data import Dataset, DataLoader
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            get_linear_schedule_with_warmup,
        )

        logger.info("Training evidence classifier...")

        # Filter evidence samples
        evidence_samples = [
            s for s in train_data
            if s.get("domain") == "evidence" and s.get("expected_label")
        ]
        if len(evidence_samples) < 10:
            logger.warning(f"Not enough evidence samples: {len(evidence_samples)}")
            return False

        label_map = {"supports": 0, "refutes": 1, "neutral": 2}
        id2label = {v: k for k, v in label_map.items()}

        # Load tokenizer and model
        model_name = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=3, id2label=id2label, label2id=label_map
        )

        # Create dataset
        class EvidenceDataset(Dataset):
            def __init__(self, samples, tokenizer, max_length=256):
                self.samples = samples
                self.tokenizer = tokenizer
                self.max_length = max_length

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                s = self.samples[idx]
                text = s.get("input_text", "")
                evidence = s.get("evidence", [])
                premise = evidence[0] if evidence else text
                hypothesis = text

                encoding = self.tokenizer(
                    premise, hypothesis,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                label = label_map.get(s["expected_label"], 2)
                return {
                    "input_ids": encoding["input_ids"].squeeze(),
                    "attention_mask": encoding["attention_mask"].squeeze(),
                    "labels": torch.tensor(label, dtype=torch.long),
                }

        train_dataset = EvidenceDataset(evidence_samples, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

        # Training setup
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        total_steps = len(train_loader) * 3  # 3 epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.train()

        # Train
        t0 = time.perf_counter()
        for epoch in range(3):
            total_loss = 0
            correct = 0
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                logits = outputs.logits

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                preds = logits.argmax(dim=-1)
                correct += (preds == batch["labels"]).sum().item()

            acc = correct / len(train_dataset)
            avg_loss = total_loss / len(train_loader)
            logger.info(f"  Epoch {epoch+1}/3: loss={avg_loss:.4f}, acc={acc:.3f}")

        elapsed = time.perf_counter() - t0
        logger.info(f"  Training completed in {elapsed:.1f}s")

        # Save model
        output_dir = _sweep_dir / "training" / "neural_models" / "evidence_classifier"
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        # Save metadata
        metadata = {
            "model": "bert-base-uncased-finetuned",
            "task": "evidence_classification",
            "labels": ["supports", "refutes", "neutral"],
            "training_examples": len(evidence_samples),
            "epochs": 3,
            "duration_seconds": elapsed,
            "status": "fine-tuned",
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"  Model saved to {output_dir}")
        return True

    except ImportError as e:
        logger.warning(f"Missing dependency for evidence classifier: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to train evidence classifier: {e}")
        return False


def train_contradiction_detector(train_data: list[dict], val_data: list[dict]) -> bool:
    """Fine-tune BERT for contradiction detection."""
    try:
        import torch
        from torch.utils.data import Dataset, DataLoader
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            get_linear_schedule_with_warmup,
        )

        logger.info("Training contradiction detector...")

        contradiction_samples = [
            s for s in train_data
            if s.get("domain") == "contradiction" and s.get("expected_label")
        ]
        if len(contradiction_samples) < 10:
            logger.warning(f"Not enough contradiction samples: {len(contradiction_samples)}")
            return False

        label_map = {"contradiction": 0, "consistent": 1, "partial": 2}
        id2label = {v: k for k, v in label_map.items()}

        model_name = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=3, id2label=id2label, label2id=label_map
        )

        class ContradictionDataset(Dataset):
            def __init__(self, samples, tokenizer, max_length=256):
                self.samples = samples
                self.tokenizer = tokenizer
                self.max_length = max_length

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                s = self.samples[idx]
                evidence = s.get("evidence", [])
                text_a = evidence[0] if len(evidence) > 0 else ""
                text_b = evidence[1] if len(evidence) > 1 else ""

                encoding = self.tokenizer(
                    text_a, text_b,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                label = label_map.get(s["expected_label"], 1)
                return {
                    "input_ids": encoding["input_ids"].squeeze(),
                    "attention_mask": encoding["attention_mask"].squeeze(),
                    "labels": torch.tensor(label, dtype=torch.long),
                }

        train_dataset = ContradictionDataset(contradiction_samples, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        total_steps = len(train_loader) * 3
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.train()

        t0 = time.perf_counter()
        for epoch in range(3):
            total_loss = 0
            correct = 0
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                logits = outputs.logits

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                preds = logits.argmax(dim=-1)
                correct += (preds == batch["labels"]).sum().item()

            acc = correct / len(train_dataset)
            avg_loss = total_loss / len(train_loader)
            logger.info(f"  Epoch {epoch+1}/3: loss={avg_loss:.4f}, acc={acc:.3f}")

        elapsed = time.perf_counter() - t0
        logger.info(f"  Training completed in {elapsed:.1f}s")

        output_dir = _sweep_dir / "training" / "neural_models" / "contradiction_detector"
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        metadata = {
            "model": "bert-base-uncased-finetuned",
            "task": "contradiction_detection",
            "labels": ["contradiction", "consistent", "partial"],
            "training_examples": len(contradiction_samples),
            "epochs": 3,
            "duration_seconds": elapsed,
            "status": "fine-tuned",
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"  Model saved to {output_dir}")
        return True

    except ImportError as e:
        logger.warning(f"Missing dependency for contradiction detector: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to train contradiction detector: {e}")
        return False


def train_intent_classifier(train_data: list[dict], val_data: list[dict]) -> bool:
    """Fine-tune BERT for intent classification."""
    try:
        import torch
        from torch.utils.data import Dataset, DataLoader
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            get_linear_schedule_with_warmup,
        )

        logger.info("Training intent classifier...")

        intent_samples = [
            s for s in train_data
            if s.get("domain") == "intent" and s.get("expected_label")
        ]
        if len(intent_samples) < 10:
            logger.warning(f"Not enough intent samples: {len(intent_samples)}")
            return False

        # Build label map from data
        labels = sorted(set(s["expected_label"] for s in intent_samples))
        label_map = {l: i for i, l in enumerate(labels)}
        id2label = {v: k for k, v in label_map.items()}

        model_name = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=len(labels), id2label=id2label, label2id=label_map
        )

        class IntentDataset(Dataset):
            def __init__(self, samples, tokenizer, max_length=128):
                self.samples = samples
                self.tokenizer = tokenizer
                self.max_length = max_length

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                s = self.samples[idx]
                text = s.get("input_text", "")

                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                label = label_map[s["expected_label"]]
                return {
                    "input_ids": encoding["input_ids"].squeeze(),
                    "attention_mask": encoding["attention_mask"].squeeze(),
                    "labels": torch.tensor(label, dtype=torch.long),
                }

        train_dataset = IntentDataset(intent_samples, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        total_steps = len(train_loader) * 3
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.train()

        t0 = time.perf_counter()
        for epoch in range(3):
            total_loss = 0
            correct = 0
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                logits = outputs.logits

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                preds = logits.argmax(dim=-1)
                correct += (preds == batch["labels"]).sum().item()

            acc = correct / len(train_dataset)
            avg_loss = total_loss / len(train_loader)
            logger.info(f"  Epoch {epoch+1}/3: loss={avg_loss:.4f}, acc={acc:.3f}")

        elapsed = time.perf_counter() - t0
        logger.info(f"  Training completed in {elapsed:.1f}s")

        output_dir = _sweep_dir / "training" / "neural_models" / "intent_classifier"
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        metadata = {
            "model": "bert-base-uncased-finetuned",
            "task": "intent_classification",
            "labels": labels,
            "training_examples": len(intent_samples),
            "epochs": 3,
            "duration_seconds": elapsed,
            "status": "fine-tuned",
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"  Model saved to {output_dir}")
        return True

    except ImportError as e:
        logger.warning(f"Missing dependency for intent classifier: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to train intent classifier: {e}")
        return False


def main():
    """Run all training."""
    print("=" * 70)
    print("SWEEP NEURAL MODEL TRAINING")
    print("=" * 70)

    # Load datasets
    train_data = load_dataset("train")
    val_data = load_dataset("val")
    logger.info(f"Loaded {len(train_data)} train, {len(val_data)} val samples")

    if not train_data:
        logger.error("No training data found. Run comprehensive_dataset.py first.")
        return

    t0 = time.perf_counter()

    # Train models
    results = {}
    results["evidence"] = train_evidence_classifier(train_data, val_data)
    results["contradiction"] = train_contradiction_detector(train_data, val_data)
    results["intent"] = train_intent_classifier(train_data, val_data)

    elapsed = time.perf_counter() - t0

    print("\n" + "=" * 70)
    print("TRAINING RESULTS")
    print("=" * 70)
    for model_name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {model_name:20s}: {status}")
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
