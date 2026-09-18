# PHASE 2 MILESTONE REPORT

Phase: Training and terminal intelligence-agent integration
Milestone: Checkpoint 7 — trained CPU intent router and terminal agent
Status: PARTIAL
Implementation:
- Reproducible multinomial Naive Bayes intelligence router with auto-generated portable JSON artifact.
- Disjoint template/entity train and test partitions, fixed seed, SHA-256 split hashes, majority baseline.
- Interactive terminal entrypoint with one-shot mode, compatible persistent file memory, extraction/summarization, and explicit unavailable OS-action responses.
Tests:
- compileall, four unit tests, fresh-start automatic training, one-shot terminal execution, memory write/recall.
Integration:
- Loaded by `sweep_neural_mesh.intelligence_agent`; memory uses companion memory version-1 schema.
Documentation: This report and metrics artifact.
Tests passed: 4 unit tests; compileall; terminal one-shot; memory write/recall.
Tests failed: 0 in final validation.
Demonstration:
- `python -m sweep_neural_mesh.intelligence_agent`
- `python -m sweep_neural_mesh.intelligence_agent --once "reconstruct the sequence of events for Operation Meridian"`
Measured results:
Baseline: 7.69% majority accuracy on 104 held-out routing examples.
Current: 99.04% (103/104) on the frozen routing holdout after paraphrase augmentation.
Target: >=90% routing accuracy plus future representative external evaluation.
Known limitations:
- Metric covers intent routing only, not general reasoning, investigation quality, or truthfulness.
- Corpus is repository-native and template-generated. FEVEROUS was reviewed but not imported because separately hosted dataset terms/files were not mounted.
- Live web collection and Device Host execution require governed production adapters.
Known failures:
- Initial router was 44.23% (46/104). One search example remains misclassified.
Dependencies satisfied: Python standard library and existing SWEEP contracts.
Dependencies remaining: license-verified external datasets, production model weights, representative benchmarks, live adapters.
Ready for dependent phases: YES for terminal routing/memory; NO for production readiness or broad accuracy claims.
