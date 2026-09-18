# SWEEP integration checkpoint 3 — isolated Device Host API router

## Scope

This checkpoint adds a **testable API router layer** for the Device Host core without yet wiring it into the live `companion.main` application. The goal is to keep the host behind structured request/response endpoints while avoiding unverified changes to the existing companion boot path.

Added:

- `companion/device_api.py`
  - `GET /api/brain/device/status`
  - `GET /api/brain/device/capabilities`
  - `POST /api/brain/device/plan`
  - `POST /api/brain/device/execute`
  - `POST /api/brain/device/approve`
  - `GET /api/brain/device/audit`
- `companion/tests/test_device_api.py`
  - route-to-host contract tests

## Why isolated first

The current sandbox cannot run the repository's full companion HTTP stack because the installed environment here does not include every declared dependency. The repository itself declares FastAPI, but this sandbox image does not provide that package. Instead of pretending the live service path was tested, this checkpoint verifies the router contract with a small `fastapi` stub that checks:

- route registration paths
- request-model validation construction
- mapping from route handler to Device Host operations
- approval flow handoff
- audit/status/capabilities responses
- traversal rejection before approval at the route boundary

This is a **contract checkpoint**, not proof that the full companion service booted here.

## Test execution

Local command executed in this run:

```sh
python3 -m unittest discover -s companion/tests -v
```

Results in the isolated device-host harness after adding the route layer:

- 27 passed
- 0 failed

Breakdown:

- 22 Device Host core + deterministic intent tests
- 5 isolated API router contract tests

## What remains before live exposure

Do not claim the host is live in SWEEP yet. The next step still requires:

1. mounting the router into `companion.main` or an equivalent authenticated entrypoint
2. adding configuration-backed host construction
3. running the repository's actual companion HTTP tests in an environment with declared dependencies installed
4. hardening the existing orchestrator/tool boundary before model-directed device execution is enabled

## Known limitations

- No real FastAPI runtime execution occurred in this sandbox.
- No user-workstation acceptance test was run for Chrome/browser/file-opening behavior.
- The router is not yet mounted into the repository's main companion app.
- Approval UI and frontend integration are still absent.
