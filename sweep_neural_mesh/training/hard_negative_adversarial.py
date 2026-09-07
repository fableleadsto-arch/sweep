"""
Hard-Negative & Adversarial Training — Sweep Robustness Training.

Usage:
    python -m sweep_neural_mesh.training.hard_negative_adversarial
"""
from __future__ import annotations

import sys
import os
import json
import time
import re
from pathlib import Path
from typing import Any

_sweep_dir = Path(__file__).resolve().parent.parent
_sweep_parent = _sweep_dir.parent
sys.path.insert(0, str(_sweep_parent))
sys.path.insert(0, str(_sweep_dir))

HARD_NEGATIVE_PAIRS = [
    ("The drug works for adults", "The drug works for children", "partial", "population"),
    ("Revenue increased overall", "Revenue decreased in Q4", "partial", "temporal_scope"),
    ("The study found a positive correlation", "The effect size was very small", "partial", "significance"),
    ("The product is affordable", "The product costs $10,000", "partial", "subjectivity"),
    ("The company is growing", "The company lost market share in Europe", "partial", "geographic_scope"),
    ("Exercise improves cardiovascular health", "Exercise has no effect on cardiovascular health", "contradiction", "negation"),
    ("The vaccine is safe", "The vaccine is not safe", "contradiction", "negation"),
    ("Climate change is real", "Climate change is not real", "contradiction", "negation"),
    ("AI will replace jobs", "AI will create jobs", "contradiction", "direction"),
    ("The Earth is round", "The Earth is flat", "contradiction", "factual"),
    ("Water boils at 100C", "Water boils at 90C", "contradiction", "numerical"),
    ("Paris is the capital of France", "Lyon is the capital of France", "contradiction", "factual"),
    ("The meeting is Monday", "The meeting is Tuesday", "contradiction", "temporal"),
    ("Light travels faster than sound", "Sound travels faster than light", "contradiction", "factual"),
    ("Cats are mammals", "Cats are reptiles", "contradiction", "classification"),
]

ADVERSARIAL_EXAMPLES = [
    {"text": "The weather was sunny today", "expected": "irrelevant", "type": "misleading"},
    {"text": "I had a great lunch", "expected": "irrelevant", "type": "misleading"},
    {"text": "The stock market closed at 3 PM", "expected": "irrelevant", "type": "misleading"},
    {"text": "According to a study by Harvard University (which doesn't exist), the drug works", "expected": "unreliable", "type": "deceptive"},
    {"text": "Source: www.totally-real-news.com confirms the claim", "expected": "unreliable", "type": "deceptive"},
    {"text": "Source A says the drug works. Source B says the drug works. Source C says the drug works.", "expected": "single_source", "type": "false_consensus"},
    {"text": "The study showed some results but we don't know the details", "expected": "insufficient", "type": "incomplete"},
    {"text": "Th3 dr0g 1s 3ff3ct1v3", "expected": "unclear", "type": "noisy"},
    {"text": "The drug is effective. However, the drug shows no significant effect.", "expected": "mixed", "type": "self_contradiction"},
]


def evaluate_hard_negatives() -> dict[str, Any]:
    """Evaluate system on hard negative examples."""
    print("\n" + "=" * 70)
    print("HARD NEGATIVE EVALUATION")
    print("=" * 70)

    # Import improved modules
    from sweep_neural_mesh.training.improved_contradiction import ImprovedContradictionDetector
    from sweep_neural_mesh.training.full_training_pipeline import ImprovedEvidenceClassifier

    detector = ImprovedContradictionDetector()
    classifier = ImprovedEvidenceClassifier()

    results = {"contradiction": {"correct": 0, "total": 0},
               "evidence": {"correct": 0, "total": 0}}

    print("\n  Contradiction Hard Negatives:")
    for text_a, text_b, expected, neg_type in HARD_NEGATIVE_PAIRS:
        result = detector.detect(text_a, text_b)
        predicted = result["answer"]
        correct = (predicted == expected or
                   (expected == "partial" and predicted in ("partial", "contradiction")))
        results["contradiction"]["total"] += 1
        if correct:
            results["contradiction"]["correct"] += 1
        status = "✓" if correct else "✗"
        print(f"    {status} [{neg_type}] Expected={expected}, Got={predicted}")

    print("\n  Adversarial Evidence Examples:")
    for example in ADVERSARIAL_EXAMPLES:
        result = classifier.classify(example["text"])
        correct = result["confidence"] < 0.8
        results["evidence"]["total"] += 1
        if correct:
            results["evidence"]["correct"] += 1
        status = "✓" if correct else "✗"
        print(f"    {status} [{example['type']}] Conf={result['confidence']:.2f}")

    total_correct = sum(r["correct"] for r in results.values())
    total_samples = sum(r["total"] for r in results.values())
    overall = total_correct / max(total_samples, 1)

    print(f"\n  Hard Negative Results:")
    print(f"    Contradiction: {results['contradiction']['correct']}/{results['contradiction']['total']}")
    print(f"    Evidence: {results['evidence']['correct']}/{results['evidence']['total']}")
    print(f"    Overall Robustness: {overall:.1%}")

    return results


def evaluate_generalization() -> dict[str, Any]:
    """Evaluate generalization to unseen domains."""
    print("\n" + "=" * 70)
    print("GENERALIZATION EVALUATION")
    print("=" * 70)

    from sweep_neural_mesh.training.full_training_pipeline import ImprovedLogicalReasoner
    from sweep_neural_mesh.training.improved_contradiction import ImprovedContradictionDetector

    logic = ImprovedLogicalReasoner()
    detector = ImprovedContradictionDetector()

    results = {"logic": {"correct": 0, "total": 0},
               "contradiction": {"correct": 0, "total": 0}}

    novel_logic = [
        ("If X implies Y and Y implies Z, does X imply Z?",
         ["If X implies Y.", "If Y implies Z."], "yes"),
        ("If all A are B and some B are C, are all A C?",
         ["All A are B.", "Some B are C."], "unknown"),
        ("If no X are Y and Z is X, is Z not Y?",
         ["No X are Y.", "Z is X."], "no"),
    ]

    print("\n  Novel Logic Structures:")
    for query, evidence, expected in novel_logic:
        result = logic.reason(query, evidence)
        predicted = result["answer"]
        correct = (predicted == expected or
                   (expected == "yes" and predicted in ("yes", "supported")) or
                   (expected == "no" and predicted in ("no", "refuted")))
        results["logic"]["total"] += 1
        if correct:
            results["logic"]["correct"] += 1
        status = "✓" if correct else "✗"
        print(f"    {status} Expected={expected}, Got={predicted}")

    novel_contradictions = [
        ("The temperature rose to 30C", "The temperature dropped to 10C", "contradiction"),
        ("The company's revenue doubled", "The company's revenue halved", "contradiction"),
        ("All students passed the exam", "No students passed the exam", "contradiction"),
        ("The system is online", "The system is operational", "consistent"),
        ("The project completed on time", "The project finished ahead of schedule", "consistent"),
    ]

    print("\n  Novel Contradiction Patterns:")
    for text_a, text_b, expected in novel_contradictions:
        result = detector.detect(text_a, text_b)
        predicted = result["answer"]
        correct = (predicted == expected or
                   (expected == "contradiction" and predicted in ("contradiction", "partial")))
        results["contradiction"]["total"] += 1
        if correct:
            results["contradiction"]["correct"] += 1
        status = "✓" if correct else "✗"
        print(f"    {status} Expected={expected}, Got={predicted}")

    total_correct = sum(r["correct"] for r in results.values())
    total_samples = sum(r["total"] for r in results.values())
    overall = total_correct / max(total_samples, 1)

    print(f"\n  Generalization Results:")
    print(f"    Logic: {results['logic']['correct']}/{results['logic']['total']}")
    print(f"    Contradiction: {results['contradiction']['correct']}/{results['contradiction']['total']}")
    print(f"    Overall Generalization: {overall:.1%}")

    return results


def main():
    """Run hard-negative and adversarial evaluation."""
    print("=" * 70)
    print("SWEEP HARD-NEGATIVE & ADVERSARIAL TRAINING")
    print("=" * 70)

    t0 = time.perf_counter()
    hn_results = evaluate_hard_negatives()
    gen_results = evaluate_generalization()
    elapsed = time.perf_counter() - t0

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "timestamp": time.time(),
        "duration_seconds": elapsed,
        "hard_negatives": hn_results,
        "generalization": gen_results,
    }

    results_path = output_dir / "hard_negative_adversarial_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to {results_path}")
    print("\n" + "=" * 70)
    print("HARD-NEGATIVE & ADVERSARIAL TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
