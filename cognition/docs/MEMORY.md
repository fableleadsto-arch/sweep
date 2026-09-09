# Sweep Cognition — Memory

Sweep's memory is **independent of the LLM context window**. Provided by
`cognition/memory.py`, backed by `CognitiveStore` (append-only JSONL).

## Working memory

`WorkingMemory(capacity=64)`

- Bounded working set for the current task/run.
- Implements keyed stores with LRU eviction when capacity is exceeded.
- Transient by design.

## Long-term memory

`LongTermMemory`

- Persistent across processes/sessions.
- Entries carry provenance: content, kind, source task, timestamp, evidence
  ids, retrieval count, last retrieved time.
- Retrieval is substring/word-overlap scored; retrieval hits are counted and
  persisted so history is retained.

## Memory kinds

| Kind         | Meaning                                  |
|--------------|------------------------------------------|
| `FACT`       | a known fact                             |
| `EVENT`      | a past event                             |
| `CONCLUSION` | a result produced by the cognitive loop  |
| `STRATEGY`   | a learned approach / plan                |
| `OBSERVATION`| a raw observed fact                      |

## Handoff across sessions

```
session A: memory.observe("x")   -> working + LTM
session B: CognitiveMemory(store_path=same_dir)
           memory.retrieve("x")  -> persists, retrieved
```

`CognitiveMemory` is the facade wiring working + long-term memory together.

## Backing store

Memory entries are stored through `store._memory_records()` with a `payload`
envelope so loading on restart is lossless and deterministic
(see `test_regression_memory_handoff_zero_loss`).

## Guarantees

- Retrieval updates retrieval statistics *on the persisted entry* (`_update=True`).
- No context-window dependence: a fresh process recalls whatever a previous
  process stored.