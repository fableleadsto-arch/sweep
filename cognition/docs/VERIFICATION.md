# Sweep Cognition — Verification

`cognition/verification.py` verifies a conclusion through **independent
paths** — neural output, logic derivation, and evidence — and combines their
verdicts.

## Qualified states

| State       | Condition                                      |
|-------------|------------------------------------------------|
| `VERIFIED`  | two or more independent checks agree           |
| `CONTESTED` | at least one independent check disagrees      |
| `UNRESOLVED`| single agreement, no agreement, or disabled checks |

## Usage

```python
eng = VerificationEngine()
eng.register_check("neural",   lambda claim_id: ...)
eng.register_check("logic",    lambda claim_id: ...)
eng.register_check("evidence", lambda claim_id: ...)
result = eng.run("c-1")          # -> VerificationResult
result.status                    # VERIFIED / CONTESTED / UNRESOLVED
```

- `register_check(name, fn)` — add an independent verdict (True=agree).
- `run(claim_id)` — executes every registered check; a raised exception marks
  the check `unrun` (degradation, not failure).
- `result(claim_id)` — returns the stored verdict.
- Helpers `verified(result)` / `contested(result)`.

## Invariants

- Single agreement is never enough: one check -> `UNRESOLVED`.
- Disagreement always yields `CONTESTED`, never a fabricated resolution.
- A crashing check degrades to `UNRESOLVED`; the run never raises.
- The cognitive loop's `verification` stage runs these checks and feeds the
  epistemic state in `uncertainty.py`.