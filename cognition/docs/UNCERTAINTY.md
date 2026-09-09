# Sweep Cognition — Uncertainty

`cognition/uncertainty.py` gives every claim an **explicit epistemic state**
instead of forced certainty. It distinguishes what Sweep knows, suspects,
cannot establish, and what is contested.

## States (all required by the milestone)

| State                   | Signal                                    |
|-------------------------|-------------------------------------------|
| `VERIFIED`              | >=2 supporting evidence across >=2 sources, no opposition |
| `PROBABLE`              | supporting evidence but weak independence |
| `POSSIBLE`              | nothing opposes; neutral/consistent input |
| `CONTESTED`             | both supporting and contradicting evidence |
| `UNRESOLVED`            | evidence contradicts with no support      |
| `INSUFFICIENT_EVIDENCE` | no evidence at all                        |

The mapping is deterministic and lives in `UncertaintyEngine.assess(signal)`.

## API

```python
eng = UncertaintyEngine()
s = eng.assess(EvidenceSignal(claim_id, statement,
                              supporting=2, contradicting=0,
                              source_diversity=2))
s.state        # EpistemicState
s.confidence   # scalar
s.reasons      # human-readable justification
s.to_dict()    # serializable
```

A claim's state can also be queried with `assert_state(claim_id)`.

## Guarantees

- **Never fabricate certainty**: ambiguous or empty input produces
  `INSUFFICIENT_EVIDENCE` / `POSSIBLE`, never `VERIFIED`.
- Every state carries explicit `reasons`.
- The cognitive loop feeds contradiction counts into the signal so genuinely
  contested input is never concluded confidently.
- Calibration is benchmarked (Phase 19 `uncertainty_calibration` = 100%).