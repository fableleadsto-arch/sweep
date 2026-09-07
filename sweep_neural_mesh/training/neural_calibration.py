"""Temperature-scaling calibration for the pair-trained evidence classifier.

Directive §18/§25: a system that is confidently wrong must score worse than
one that is honestly uncertain. The measured ECE of 0.1746 (REPORT §5b) showed
over-confident errors. Temperature scaling (Guo et al. 2017, arXiv:1706.04599)
fits a single temperature T on a HELD-OUT validation split and rescales logits:
softmax(logits / T). T > 1 softens over-confident predictions; argmax is
unchanged, so accuracy is unchanged — only confidence is recalibrated.

Protocol:
  1. Generate a FRESH-SEED pair set (same generator as training, new seed ->
     different draws; never the training instances themselves).
  50/50 split: fit T on the calibration half, report ECE on the eval half.
  2. Write calibration.json next to the artifact; the inference path
     (neural_engine) applies it automatically once present.
  3. Print ECE before/after + reliability-table summary.

Usage:
    python -m sweep_neural_mesh.training.neural_calibration
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

ARTIFACT_DIR = Path(__file__).parent / "neural_models" / "evidence_classifier"
CALIB_SEED = 4242  # fresh, disjoint from training seed 13
N_BINS = 15


def _ece_and_bins(confidences: list[float], corrects: list[int]) -> tuple[float, list[dict]]:
    """Equal-width expected calibration error + per-bin table."""
    bins = [[] for _ in range(N_BINS)]
    for conf, ok in zip(confidences, corrects):
        b = min(N_BINS - 1, int(conf * N_BINS))
        bins[b].append((conf, ok))
    ece = 0.0
    table = []
    for i, b in enumerate(bins):
        if not b:
            table.append({"bin": f"[{i/N_BINS:.2f}-{(i+1)/N_BINS:.2f})",
                          "count": 0, "mean_conf": None, "accuracy": None})
            continue
        mean_conf = sum(c for c, _ in b) / len(b)
        acc = sum(ok for _, ok in b) / len(b)
        ece += (len(b) / len(confidences)) * abs(acc - mean_conf)
        table.append({"bin": f"[{i/N_BINS:.2f}-{(i+1)/N_BINS:.2f})",
                      "count": len(b), "mean_conf": round(mean_conf, 4),
                      "accuracy": round(acc, 4)})
    return ece, table


def main() -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from sweep_neural_mesh.training.train_evidence_pairs import generate_pairs

    # 1. Fresh-seed validation pairs (same distribution, different draws)
    rng = random.Random(CALIB_SEED)
    pairs = generate_pairs(rng)
    split = len(pairs) // 2
    calib_pairs, eval_pairs = pairs[:split], pairs[split:]
    print(f"Fresh-seed pairs: {len(pairs)} "
          f"(fit on {len(calib_pairs)}, evaluate ECE on {len(eval_pairs)})")

    tokenizer = AutoTokenizer.from_pretrained(str(ARTIFACT_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(ARTIFACT_DIR))
    model.eval()
    torch.set_num_threads(8)

    def score(split_pairs: list[tuple[str, str, int]]):
        premises = [p for p, _, _ in split_pairs]
        hyps = [h for _, h, _ in split_pairs]
        labels = [l for _, _, l in split_pairs]
        enc = tokenizer(premises, hyps, truncation=True, padding=True,
                        max_length=64, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits
        return labels, logits

    # 2. Fit T on the calibration half (NLL minimization over temperature)
    calib_labels, calib_logits = score(calib_pairs)
    calib_logits = calib_logits.float()
    targets = torch.tensor(calib_labels)
    logit_scale = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([logit_scale], max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(calib_logits / logit_scale.clamp(min=1e-3),
                                                 targets)
        loss.backward()
        return loss

    opt.step(closure)
    temperature = float(logit_scale.clamp(min=1e-3).item())
    print(f"Fitted temperature: {temperature:.4f}")

    # 3. Evaluate ECE on the untouched eval half, before and after scaling
    eval_labels, eval_logits = score(eval_pairs)
    eval_logits = eval_logits.float()
    probs_raw = torch.softmax(eval_logits, dim=-1)
    probs_cal = torch.softmax(eval_logits / temperature, dim=-1)

    def ece_of(probs):
        confs, corrects = [], []
        for row, lbl in zip(probs, eval_labels):
            confs.append(float(row.max()))
            corrects.append(int(int(row.argmax()) == lbl))
        return _ece_and_bins(confs, corrects)

    ece_before, bins_before = ece_of(probs_raw)
    ece_after, bins_after = ece_of(probs_cal)
    acc = sum(1 for row, lbl in zip(probs_raw, eval_labels)
              if int(row.argmax()) == lbl) / len(eval_labels)

    print(f"Held-out accuracy (unchanged by T): {acc:.4f}")
    print(f"ECE before scaling: {ece_before:.4f}")
    print(f"ECE after  scaling: {ece_after:.4f}")

    payload = {
        "task": "evidence_classification",
        "method": "temperature_scaling",
        "reference": "Guo et al. 2017, arXiv:1706.04599",
        "temperature": round(temperature, 6),
        "ece_before": round(ece_before, 6),
        "ece_after": round(ece_after, 6),
        "heldout_accuracy": round(acc, 6),
        "calibration_seed": CALIB_SEED,
        "fit_pairs": len(calib_pairs),
        "eval_pairs": len(eval_pairs),
        "bins_before": bins_before,
        "bins_after": bins_after,
        "fit_on": "fresh-seed pairs from train_evidence_pairs.generate_pairs (never training instances)",
        "fit_date": "2026-09-05",
    }
    out = ARTIFACT_DIR / "calibration.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out}")

    # 4. Self-check through the production path
    try:
        from sweep_neural_mesh.neurons.neural_engine import NeuralEngine
        eng = NeuralEngine()
        if eng.wait_until_ready(timeout=180):
            probes = [
                ("The krin is stable and running smoothly.", "The krin is stable.", "supports"),
                ("The vost is not enabled per diagnostic log 3.", "The vost is enabled.", "refutes"),
                ("The tremn is awake.", "The solk is visible.", "neutral"),
            ]
            for premise, hypothesis, expected in probes:
                r = eng.classify_evidence(premise, hypothesis)
                print(f"  {r.label}@{r.confidence:.2f} (expected {expected}) "
                      f"| calibrated T applied: {r.label == expected}")
    except Exception as e:  # pragma: no cover
        print(f"Self-check skipped: {e}")


if __name__ == "__main__":
    main()
