# SWEEP integration checkpoint 2 — Device Host core

## Scope

This checkpoint adds the first tested **Device Host core** under `companion/` without replacing the existing neural engine. It is intentionally isolated from the current HTTP routes and the current orchestrator tool loop because those layers still need additional hardening before real device execution should be exposed.

Implemented in this checkpoint:

- `companion/device_host.py`
  - allowlisted actions only
  - disabled-by-default host
  - explicit risk levels
  - confirmation/approval flow
  - persistent allow/deny decisions
  - dry-run mode
  - simulation mode
  - audit log
  - path-root enforcement
  - alias-root traversal rejection
  - no shell execution
- `companion/device_intents.py`
  - deterministic parsing of a small set of direct device requests into structured action plans
- tests for hostile parameters, approval flow, persistence, dry-run and simulation

Not implemented in this checkpoint:

- live FastAPI endpoints
- UI / control panel
- orchestrator exposure
- browser automation beyond default URL opening
- screenshots, clipboard, timers, running-app inspection, focus/close app control
- Windows/macOS acceptance execution
- multi-step observation memory inside the neural mesh

## Security model

The host treats action requests as untrusted input. It does **not** trust model-supplied risk or confirmation claims.

Controls implemented here:

- action allowlist
- strict application-name validation
- URL scheme restriction (`http`/`https` only)
- allowed-root path enforcement
- alias-bound path enforcement (`alias:downloads/../...` is rejected)
- approvals expire
- permanent deny persists
- audit records omit file contents and secrets
- no `shell=True`, no arbitrary command execution API

A direct test failure exposed an approval-before-validation bug during development; the final checkpoint fixes it so traversal payloads are rejected **before** approval requests are created.

## Implemented actions

Current action registry:

- `open_application`
- `open_url`
- `open_file`
- `open_folder`
- `create_folder`
- `list_directory`
- `search_local_files`
- `get_system_information`

These are the only implemented actions. Unsupported actions fail closed.

## Planning bridge

`plan_device_actions()` currently handles a narrow deterministic subset:

- `Open Chrome.`
- `Open Chrome and go to example.com`
- `Open my Downloads folder`
- `Go to https://example.com/path`
- `search local files for budget in downloads`
- `create a folder called Research in downloads`

If the folder location is omitted, planning returns a clarification requirement instead of inventing a destination.

## Test execution

Local targeted command used in this run:

```sh
python3 -m unittest discover -s companion/tests -v
```

Executed tests in the isolated device-host harness:

- 22 passed
- 0 failed after the alias-boundary fix

Coverage in this checkpoint includes:

- disabled-by-default behavior
- capability/status reporting
- approval requirement for risky actions
- allow-once / always-allow / permanently-deny behavior
- persistence across host restart
- dry-run no-op behavior
- real local folder creation under an allowed root
- path traversal rejection
- javascript URL rejection
- command-injection application name rejection
- directory listing and filename search
- system information without confirmation
- audit logging without file contents
- expired approvals
- deterministic intent parsing for basic device requests

## Known limitations

- This is a **core module checkpoint**, not a finished end-to-end feature.
- Actual GUI application launch, default-browser open and path-opening behavior are only implemented as best-effort adapter calls and were **not** acceptance-tested on a user workstation in this run.
- The current orchestrator/tool registry needs additional schema/result hardening before real device tools should be exposed through model-directed execution.
- No claim is made that SWEEP can already open Chrome end-to-end from the current UI.
- The neural mesh does not yet consume persisted action observations from this checkpoint.

## Next recommended step

Integrate this core behind authenticated companion endpoints and keep the first exposed mode limited to:

1. status/capabilities
2. deterministic plan preview
3. dry-run execution
4. simulation execution
5. approval-backed real execution only after the companion tool/result boundary is hardened
