PHASE 0 MILESTONE REPORT
Phase:
Device Host pre-exposure hardening
Milestone:
Checkpoint 4 — companion orchestrator execution boundaries
Status: PARTIAL
Implementation:
Added execution-side hardening for the companion orchestrator before Device Host exposure. Tool execution now rejects non-object planned input, rejects schema-invalid planned input before tool code runs, enforces the request tool allowlist during execution rather than only during tool advertisement, and caps the number of tool steps executed from a single model-produced batch.
Tests:
A focused local harness for the orchestrator hardening logic was executed with five passing tests.
Integration:
The repository orchestrator and a focused repository test file were updated on the development branch. The repository's full pytest/FastAPI companion test stack was not executed in this sandbox.
Documentation:
This report documents the checkpoint intent, results, and remaining blockers.
Tests passed:
5 focused orchestrator hardening tests in the local harness.
Tests failed:
0 in the local hardening harness.
Demonstration:
Validated rejection of missing required fields, wrong input types, non-object tool input, hidden tool execution through the whitelist boundary, and oversized single-plan tool batches.
Measured results:
Baseline:
The companion orchestrator advertised a tool whitelist but did not enforce it at execution time, accepted arbitrary planned input shapes until tool code failed, and could execute an unbounded number of steps from a single model-produced batch.
Current:
The updated branch design blocks those three classes of behavior and adds focused tests for them.
Target:
Run the repository companion tests in an environment with declared dependencies installed, then wire Device Host behind authenticated companion endpoints.
Known limitations:
This sandbox does not include the repository's full HTTP test dependency set, so this checkpoint does not claim end-to-end companion service execution.
Known failures:
None in the focused hardening harness.
Dependencies satisfied:
Focused hardening logic and tests are implemented and pushed to the development branch.
Dependencies remaining:
Run the real repository companion tests and then wire the Device Host router into the live companion app.
Ready for dependent phases: YES
