PHASE 2 MILESTONE REPORT
Phase: Terminal and persistent memory integration
Milestone: Checkpoint 13 — production MemoryService adapter and installable CLI
Status: PARTIAL
Implementation:
- Replaced duplicate terminal file-memory logic with a synchronous adapter over companion.memory.MemoryService.
- Preserved CompatibleFileMemory as an import alias only.
- Added remember, recall, status, user, and workspace terminal options.
- Added the sweep console entry point.
- Corrected package discovery to include companion and sweep_neural_mesh.
- Declared qdrant-client, which production memory imports.
Tests: Stubbed production-service delegation test plus CI workflow.
Integration: Uses BrainSettings, MemoryService, file fallback, optional Qdrant, existing router, SweepAPI adapter, and Device Host planner.
Documentation: This report.
Tests passed: 1/1 local adapter test; branch CI pending.
Tests failed: 0 local focused tests.
Demonstration: sweep --status; sweep --remember TEXT; sweep --recall QUERY; sweep --once TEXT --neural.
Measured results: Production adapter calls verified with user/workspace forwarding.
Baseline: Terminal duplicated persistence and package discovery excluded core runtime packages.
Current: One production memory implementation and installable console command.
Target: Dependency-complete editable-install and terminal smoke test on a network-enabled runner.
Known limitations: Sync adapter intentionally rejects use inside an active event loop; device execution remains plan-only.
Known failures: Full dependency installation not tested locally due sandbox DNS.
Dependencies satisfied: Existing production memory contract and compatible on-disk schema.
Dependencies remaining: CI result and dependency-complete package-install smoke test.
Ready for dependent phases: NO until CI passes.
