"""Phase 5 — Inspectable Rule-Based Reasoning.

Rules are explicit, versioned objects. Every rule that fires records a
trace entry, so the exact rule responsible for a derived result can be
identified after execution.

Rule matching: rules have `if_all` premises (each a predicate on the working
facts) and produce `then` conclusions. The engine applies all matching rules;
results (including which rule fired, and its version) are recorded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .ids import content_id


@dataclass
class Rule:
    """A versioned, traceable rule."""

    rule_id: str
    version: str
    description: str = ""
    if_all: List[Callable[[Dict[str, Any]], bool]] = field(default_factory=list)
    then: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def make(cls, rule_id: str, version: str, description: str,
             conditions: List[Callable[[Dict[str, Any]], bool]],
             action: Callable[[Dict[str, Any]], Dict[str, Any]],
             enabled: bool = True) -> "Rule":
        return cls(rule_id=rule_id, version=version, description=description,
                   if_all=conditions, then=action, enabled=enabled)


@dataclass
class RuleTrace:
    """A record showing a rule fired and what it derived."""

    rule_id: str
    rule_version: str
    description: str
    input_facts: Dict[str, Any]
    output: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_version": self.rule_version,
            "description": self.description,
            "input": self.input_facts,
            "output": self.output,
        }


class RuleEngine:
    """Executes explicit, versioned rules and records a full trace."""

    def __init__(self):
        self._rules: List[Rule] = []
        self._trace: List[RuleTrace] = []
        # for deterministic ordering
        self._rule_index: Dict[str, int] = {}

    def add_rule(self, rule: Rule) -> None:
        if rule.rule_id in self._rule_index:
            self._rules[self._rule_index[rule.rule_id]] = rule
        else:
            self._rule_index[rule.rule_id] = len(self._rules)
            self._rules.append(rule)

    @property
    def rules(self) -> List[Rule]:
        return list(self._rules)

    def trace(self) -> List[RuleTrace]:
        return list(self._trace)

    def clear_trace(self) -> None:
        self._trace = []

    def _get_rule(self, rule_id: str) -> Optional[Rule]:
        idx = self._rule_index.get(rule_id)
        if idx is None:
            return None
        return self._rules[idx]

    def run(self, facts: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all matching rules against facts; return the derived result.

        The trace records every rule that fired, its version, and its output,
        so any derived result can be attributed to a specific rule.
        """
        self.clear_trace()
        working = dict(facts)
        # deterministic order (insertion order)
        for rule in list(self._rules):
            if not rule.enabled:
                continue
            if all(cond(working) for cond in rule.if_all):
                out = rule.then(working)
                derived_id = content_id(rule.rule_id, str(rule.version), str(out))
                out = {"_derived": derived_id, "_rule": rule.rule_id, **out}
                working.update(out)
                self._trace.append(RuleTrace(
                    rule_id=rule.rule_id,
                    rule_version=rule.version,
                    description=rule.description,
                    input_facts=_scrub(working),
                    output=out,
                ))
        return working

    def explain(self, result_key: str) -> Optional[Rule]:
        """Identify the exact rule responsible for a result key, if any."""
        # find the last trace that produced the given result key
        for t in reversed(self._trace):
            if result_key in t.output:
                return self._get_rule(t.rule_id)
        return None


def _scrub(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def _counts_greater(key: str, threshold: int) -> Callable[[Dict[str, Any]], bool]:
    def cond(facts: Dict[str, Any]) -> bool:
        return int(facts.get(key, 0)) > threshold
    return cond


def _key_equals(key: str, value: Any) -> Callable[[Dict[str, Any]], bool]:
    def cond(facts: Dict[str, Any]) -> bool:
        return facts.get(key) == value
    return cond


def _affirmative(answer_key: str) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    def action(facts: Dict[str, Any]) -> Dict[str, Any]:
        return {"conclusion": facts.get(answer_key, ""), "verdict": "supported"}
    return action


def _regex_match(input_key: str, bodes_well: bool) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    def action(facts: Dict[str, Any]) -> Dict[str, Any]:
        return {"derived": "regex-applied", "input": facts.get(input_key, "")}
    return action


def default_compact_ruleset() -> RuleEngine:
    """A compact, deterministic ruleset used in tests and demos."""
    eng = RuleEngine()
    def cond_has(base: str) -> Callable[[Dict[str, Any]], bool]:
        return lambda f: bool(f.get(base))
    def act_victory(f: Dict[str, Any]) -> Dict[str, Any]:
        return {"verdict": "victory", "winner": f.get("pieces_at_goal", "")}
    # rule: victory if player at goal and time remaining
    eng.add_rule(Rule.make(
        rule_id="victory_if_at_goal",
        version="1.0.0",
        description="A player having moved all pieces to the goal wins",
        conditions=[_counts_greater("pieces_at_goal", 0), _key_equals("time_remaining", "yes")],
        action=act_victory,
    ))
    # rule: alert on high pressure
    def cond_high(f: Dict[str, Any]) -> bool:
        return float(f.get("pressure", 0)) > 1.0
    def act_alert(f: Dict[str, Any]) -> Dict[str, Any]:
        return {"alert": "high-pressure", "level": f.get("pressure")}
    eng.add_rule(Rule.make(
        rule_id="alert_on_high_pressure", version="2.1.0", description="Raise alarm when pressure > 1.0",
        conditions=[cond_high], action=act_alert,
    ))
    return eng