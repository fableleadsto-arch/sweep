PHASE 2 MILESTONE REPORT
Phase: Device Host live exposure
Milestone: Checkpoint 10 — authenticated disabled-by-default live route mount
Status: PARTIAL
Implementation:
- Added environment-backed Device Host settings.
- Mounted the existing device router through the app extension convention.
- Applied the existing bearer-token dependency to every device endpoint.
- Defaults: disabled, non-simulated, dry-run, no allowed roots, empty app allowlist.
Tests: Secure defaults, explicit JSON configuration, malformed configuration fail-closed, compileall.
Integration: Reuses DeviceHost, DeviceHostConfig, device_api, device_intents, main app, and require_token.
Documentation: This report.
Tests passed: 3/3 focused settings tests plus compileall; prior Device Host core/router tests remain 27/27 in their validated harness.
Tests failed: 0 focused final tests.
Demonstration: GET /api/brain/device/status after starting companion; real actions remain disabled unless explicitly configured.
Measured results: Route configuration is fail-closed; no real OS action was executed.
Baseline: Device router existed but was not mounted in the live app.
Current: Router mounts with authentication and secure defaults.
Target: Dependency-complete FastAPI integration tests and authorized end-to-end OS acceptance tests.
Known limitations: Full FastAPI suite was not executable in the sandbox; real OS actions not tested here.
Known failures: None in focused tests.
Dependencies satisfied: Existing permission manager, audit log, action registry, deterministic planner, bearer auth.
Dependencies remaining: Production host configuration, UI approval flow, Windows/macOS end-to-end tests.
Ready for dependent phases: YES for simulation/dry-run integration; NO for unrestricted real execution.
