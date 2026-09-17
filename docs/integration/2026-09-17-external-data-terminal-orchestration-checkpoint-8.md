PHASE 2 MILESTONE REPORT
Phase: External data governance and terminal orchestration
Milestone: Checkpoint 8 — pinned CLINC150 import, BIG-bench evaluation lock, SweepAPI terminal adapter
Status: PARTIAL
Implementation:
- Pinned CLINC150 data_full.json to upstream revision/blob, CC BY 3.0, preserving upstream splits.
- Added validating CLINC150 normalizer that rejects unpinned production inputs.
- Pinned BIG-bench logical deduction as evaluation-only and rejects training because its canary prohibits it.
- Added terminal adapters for SweepAPI observations and Device Host plan-only previews.
- Added `python -m sweep_neural_mesh.terminal --neural`.
Tests: Source policy, split preservation, neural success/unavailable propagation, terminal neural routing, Device Host non-execution.
Integration: Reuses IntelligenceRouter, compatible memory, SweepAPI, and device_intents.
Documentation: This report and external_source_lock.json.
Tests passed: 6/6 combined unit tests plus compileall.
Tests failed: 0 in final combined validation.
Demonstration: `python -m sweep_neural_mesh.terminal --neural --once "evaluate the evidence"`
Measured results:
Baseline: External sources were only unverified catalog entries.
Current: Two exact upstream revisions are usage-locked; CLINC importer and terminal adapters tested with fixtures.
Target: Import the complete 2,495,390-byte pinned CLINC file and run external held-out evaluation.
Known limitations: Full CLINC file could not be copied into this sandbox through the GitHub API; no external training score is claimed.
Known failures: None in final tests.
Dependencies satisfied: Inspectable license, revision, blob ID, source size, preserved split contract.
Dependencies remaining: Mount/download full pinned CLINC file; production inference dependencies and live Device Host approvals.
Ready for dependent phases: YES
