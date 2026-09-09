# Sweep Cognition — Rule Engine

`cognition/rules.py` is an **inspectable, versioned** rule engine. Every rule
is an explicit object; every firing is recorded in a trace.

## Rule model

```
Rule(rule_id, version, description, if_all=[conditions], then=action, enabled)
```

- `conditions` are predicates over the working facts dict.
- `action` produces an output dict.
- Rules are added/updated by `rule_id` (`add_rule` keeps deterministic order).

## Firing and trace

When a rule fires, a `RuleTrace(rule_id, rule_version, description,
input_facts, output)` is recorded. After execution you can recover exactly
which rule (and which version) produced a given result. Trace entries
serialize via `to_dict`.

## Rule management

- `add_rule(rule)` — insert or replace by `rule_id` (version tracked).
- `rules` property returns the ordered rule list.
- Disabled rules (`enabled=False`) are not fired.
- Versioning: updating a rule creates a newer-version replacement while the
  trace still records which version actually fired.

## Uses

- Default compact ruleset for recurring reasoning patterns.
- `feedback.py` can attach an `ADD_RULE` learning action when a rule gap is
  classified — the new rule becomes inspectable in the trace
  (see `test_feedback_*` for the feedback loop wiring).