PHASE 2 MILESTONE REPORT
Phase: External dataset training and evaluation
Milestone: Checkpoint 12 — decontaminated CLINC150 OOS-calibrated evaluation
Status: PARTIAL
Implementation:
- Removed five exact train/evaluation duplicate texts before fitting.
- Trained TF-IDF plus LinearSVC on 14,995 in-scope training examples.
- Selected an OOS rejection threshold on validation only.
- Applied the frozen threshold once to the untouched 5,500-example test/OOS split.
Tests:
- Pinned dataset Git blob verified.
- Upstream split counts verified and preserved.
- Post-filter exact-overlap audit is zero across train/validation/test.
- GitHub Actions train-and-evaluate job completed successfully.
Integration: Model artifact retained by GitHub Actions for 30 days; metrics committed under docs/metrics.
Documentation: This report and docs/metrics/clinc150_metrics.json.
Tests passed: Dataset verification, normalization, training, validation calibration, untouched test evaluation, artifact upload.
Tests failed: No runtime failure; target performance remains unmet.
Demonstration: Re-run the CLINC150 pinned training workflow.
Measured results:
- Validation accuracy 82.84%; macro-F1 87.50%; in-scope 82.70%; OOS recall 87.00%.
- Test accuracy 83.60%; macro-F1 87.21%; in-scope 83.04%; OOS recall 86.10%.
- Balanced test in-scope/OOS score 84.57%.
- Training time 1.77 seconds on GitHub Linux/Python 3.12/scikit-learn 1.9.1.
Baseline: v1 test accuracy 76.47%, macro-F1 82.99%, in-scope 91.02%, OOS recall 11.00%, five exact train/evaluation overlaps.
Current: v2 test accuracy 83.60%, macro-F1 87.21%, in-scope 83.04%, OOS recall 86.10%, zero exact overlap.
Target: Representative externally held-out routing performance at or above an agreed production threshold without sacrificing safety/OOS rejection.
Known limitations: Threshold optimization trades in-scope recall for OOS safety; no confidence intervals, near-duplicate audit, or cross-dataset generalization yet.
Known failures: Broad 90% target not met; model is not approved as a general neural-mesh or production-accuracy claim.
Dependencies satisfied: License, attribution, checksum, immutable splits, seed, environment capture, exact-overlap decontamination, validation-only calibration.
Dependencies remaining: Bootstrap confidence intervals, near-duplicate audit, independent paraphrase evaluation, calibration metrics, downstream router integration gate.
Ready for dependent phases: YES for research comparison and shadow-mode integration; NO for production default routing.
