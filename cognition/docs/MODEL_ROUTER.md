# Sweep Cognition — Model Router

`cognition/router.py` selects an appropriate model / deterministic tool /
reasoning path for each task and **logs every routing decision**.

## Task -> route mapping

| Task class        | Route selected |
|-------------------|----------------|
| arithmetic        | arithmetic -> calculator |
| OCR               | OCR / `ocr`            |
| visual perception | vision                 |
| simple logic      | simple_logic / logic engine |
| complex reasoning | reasoning model        |
| failed model      | fallback (chain)       |

## Behavior

- `classify(task)` returns the deterministic route name using regex patterns.
- `route(task)` records a `RouterDecision` (task, selected, confidence,
  reason, timestamp, fallback_chain) into `log`.
- `execute(task, ctx)` runs the selected handler; if it raises, it walks the
  fallback chain (`_fallback_chain`) and appends fallback choices to the
  decision log. If every route fails, a `RuntimeError` is raised — never a
  silent success.
- `decisions(task=None)` / `stats()` expose the routing audit trail.

## Handlers

`Router.register(name, handler)` and `CognitiveLoop.register_neural(name, ...)`
both wire handlers into the map; the loop's `neural_reasoning` stage runs them.

## Guarantees

- Every routing decision is logged and inspectable.
- A failed model triggers a logged fallback before any answer is produced.
- Deterministic classification: same task string -> same route.