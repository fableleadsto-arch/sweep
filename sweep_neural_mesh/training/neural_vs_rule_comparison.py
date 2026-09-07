"""
Neural vs Rule-Based Comparison — Sweep Full-System Training.

Compares old rule-based system with new neural models on the same test set.
Measures actual improvement from the neural upgrade.

Usage:
    python -m sweep_neural_mesh.training.neural_vs_rule_comparison
"""
from __future__ import annotations

import sys
import os
import json
import time
from pathlib import Path
from typing import Any

_sweep_dir = Path(__file__).resolve().parent.parent
_sweep_parent = _sweep_dir.parent
sys.path.insert(0, str(_sweep_parent))
sys.path.insert(0, str(_sweep_dir))


def run_comparison():
    """Run comparison between rule-based and neural systems."""
    print("=" * 70)
    print("NEURAL vs RULE-BASED COMPARISON")
    print("=" * 70)
    
    t0 = time.perf_counter()
    
    # Load test data
    test_path = Path(__file__).parent / "datasets" / "comprehensive_test.jsonl"
    test_data = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line))
    
    print(f"Loaded {len(test_data)} test samples")
    
    # Initialize systems
    from sweep_neural_mesh.training.neural_integration import NeuralLayer
    from sweep_neural_mesh.training.improved_contradiction import ImprovedContradictionDetector
    from sweep_neural_mesh.training.full_training_pipeline import (
        ImprovedLogicalReasoner, ImprovedEvidenceClassifier, ImprovedTemporalReasoner
    )
    
    neural = NeuralLayer()
    rule_contradiction = ImprovedContradictionDetector()
    rule_logic = ImprovedLogicalReasoner()
    rule_evidence = ImprovedEvidenceClassifier()
    rule_temporal = ImprovedTemporalReasoner()
    
    # Results
    results = {
        "evidence": {"rule": {"correct": 0, "total": 0}, "neural": {"correct": 0, "total": 0}},
        "contradiction": {"rule": {"correct": 0, "total": 0}, "neural": {"correct": 0, "total": 0}},
        "logic": {"rule": {"correct": 0, "total": 0}, "neural": {"correct": 0, "total": 0}},
        "temporal": {"rule": {"correct": 0, "total": 0}, "neural": {"correct": 0, "total": 0}},
    }
    
    # Test evidence classification
    print("\n--- Evidence Classification ---")
    for sample in [s for s in test_data if s["domain"] == "evidence"]:
        expected = sample["expected_label"]
        text = sample["input_text"]
        
        # Rule-based
        rule_result = rule_evidence.classify(text)
        rule_correct = rule_result["answer"] == expected
        results["evidence"]["rule"]["total"] += 1
        if rule_correct:
            results["evidence"]["rule"]["correct"] += 1
        
        # Neural
        neural_result = neural.classify_evidence(text)
        neural_correct = neural_result["answer"] == expected
        results["evidence"]["neural"]["total"] += 1
        if neural_correct:
            results["evidence"]["neural"]["correct"] += 1
    
    rule_acc = results["evidence"]["rule"]["correct"] / max(results["evidence"]["rule"]["total"], 1)
    neural_acc = results["evidence"]["neural"]["correct"] / max(results["evidence"]["neural"]["total"], 1)
    print(f"  Rule-based: {results['evidence']['rule']['correct']}/{results['evidence']['rule']['total']} ({rule_acc:.1%})")
    print(f"  Neural:     {results['evidence']['neural']['correct']}/{results['evidence']['neural']['total']} ({neural_acc:.1%})")
    print(f"  Improvement: {(neural_acc - rule_acc):+.1%}")
    
    # Test contradiction detection
    print("\n--- Contradiction Detection ---")
    for sample in [s for s in test_data if s["domain"] == "contradiction"]:
        expected = sample["expected_label"]
        evidence = sample.get("evidence", [])
        if len(evidence) < 2:
            continue
        
        # Rule-based
        rule_result = rule_contradiction.detect(evidence[0], evidence[1])
        rule_pred = rule_result["answer"]
        rule_correct = (rule_pred == expected or 
                       (expected == "partial" and rule_pred in ("partial", "contradiction")))
        results["contradiction"]["rule"]["total"] += 1
        if rule_correct:
            results["contradiction"]["rule"]["correct"] += 1
        
        # Neural
        neural_result = neural.detect_contradiction(evidence[0], evidence[1])
        neural_pred = neural_result["answer"]
        neural_correct = (neural_pred == expected or
                         (expected == "partial" and neural_pred in ("partial", "contradiction")))
        results["contradiction"]["neural"]["total"] += 1
        if neural_correct:
            results["contradiction"]["neural"]["correct"] += 1
    
    rule_acc = results["contradiction"]["rule"]["correct"] / max(results["contradiction"]["rule"]["total"], 1)
    neural_acc = results["contradiction"]["neural"]["correct"] / max(results["contradiction"]["neural"]["total"], 1)
    print(f"  Rule-based: {results['contradiction']['rule']['correct']}/{results['contradiction']['rule']['total']} ({rule_acc:.1%})")
    print(f"  Neural:     {results['contradiction']['neural']['correct']}/{results['contradiction']['neural']['total']} ({neural_acc:.1%})")
    print(f"  Improvement: {(neural_acc - rule_acc):+.1%}")
    
    # Test logic
    print("\n--- Logical Reasoning ---")
    for sample in [s for s in test_data if s["domain"] == "logic"]:
        expected = sample["expected_label"]
        evidence = sample.get("evidence", [])
        
        # Rule-based
        rule_result = rule_logic.reason(sample["input_text"], evidence)
        rule_pred = rule_result["answer"]
        rule_correct = (rule_pred == expected or
                       (expected == "yes" and rule_pred in ("yes", "supported")) or
                       (expected == "no" and rule_pred in ("no", "refuted")))
        results["logic"]["rule"]["total"] += 1
        if rule_correct:
            results["logic"]["rule"]["correct"] += 1
        
        # Neural (use rule-based for now since we don't have a neural logic model)
        results["logic"]["neural"]["total"] += 1
        if rule_correct:
            results["logic"]["neural"]["correct"] += 1
    
    rule_acc = results["logic"]["rule"]["correct"] / max(results["logic"]["rule"]["total"], 1)
    print(f"  Rule-based: {results['logic']['rule']['correct']}/{results['logic']['rule']['total']} ({rule_acc:.1%})")
    print(f"  Neural:     {results['logic']['neural']['correct']}/{results['logic']['neural']['total']} ({rule_acc:.1%}) (no neural model yet)")
    
    # Test temporal
    print("\n--- Temporal Reasoning ---")
    for sample in [s for s in test_data if s["domain"] == "temporal"]:
        expected = sample["expected_label"]
        
        # Rule-based
        rule_result = rule_temporal.reason(sample["input_text"])
        rule_pred = rule_result["answer"]
        rule_correct = (rule_pred == expected or
                       expected in rule_pred or rule_pred in expected)
        results["temporal"]["rule"]["total"] += 1
        if rule_correct:
            results["temporal"]["rule"]["correct"] += 1
        
        # Neural (use rule-based for now)
        results["temporal"]["neural"]["total"] += 1
        if rule_correct:
            results["temporal"]["neural"]["correct"] += 1
    
    rule_acc = results["temporal"]["rule"]["correct"] / max(results["temporal"]["rule"]["total"], 1)
    print(f"  Rule-based: {results['temporal']['rule']['correct']}/{results['temporal']['rule']['total']} ({rule_acc:.1%})")
    print(f"  Neural:     {results['temporal']['neural']['correct']}/{results['temporal']['neural']['total']} ({rule_acc:.1%}) (no neural model yet)")
    
    # Overall
    elapsed = time.perf_counter() - t0
    
    total_rule = sum(r["rule"]["correct"] for r in results.values())
    total_neural = sum(r["neural"]["correct"] for r in results.values())
    total_samples = sum(r["rule"]["total"] for r in results.values())
    
    rule_overall = total_rule / max(total_samples, 1)
    neural_overall = total_neural / max(total_samples, 1)
    
    print("\n" + "=" * 70)
    print("OVERALL COMPARISON")
    print("=" * 70)
    print(f"  Rule-based: {total_rule}/{total_samples} ({rule_overall:.1%})")
    print(f"  Neural:     {total_neural}/{total_samples} ({neural_overall:.1%})")
    print(f"  Improvement: {(neural_overall - rule_overall):+.1%}")
    print(f"  Duration: {elapsed:.1f}s")
    
    # Model status
    print("\n" + "=" * 70)
    print("MODEL STATUS")
    print("=" * 70)
    status = neural.get_status()
    for model_type, method in status["model_types"].items():
        print(f"  {model_type}: {method}")
    
    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comparison_results = {
        "timestamp": time.time(),
        "duration_seconds": elapsed,
        "total_samples": total_samples,
        "rule_overall": rule_overall,
        "neural_overall": neural_overall,
        "improvement": neural_overall - rule_overall,
        "by_domain": {},
    }
    
    for domain, domain_results in results.items():
        rule_acc = domain_results["rule"]["correct"] / max(domain_results["rule"]["total"], 1)
        neural_acc = domain_results["neural"]["correct"] / max(domain_results["neural"]["total"], 1)
        comparison_results["by_domain"][domain] = {
            "rule_accuracy": rule_acc,
            "neural_accuracy": neural_acc,
            "improvement": neural_acc - rule_acc,
        }
    
    results_path = output_dir / "neural_vs_rule_comparison.json"
    with open(results_path, "w") as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_comparison()
