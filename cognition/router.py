"""Phase 7 — Task-Aware Cognitive Resource Selection (Model Router).

Selects an appropriate model / deterministic tool / reasoning path for each
task. Every routing decision is logged. Failed models trigger fallbacks.

Mapping (from the milestone):
  arithmetic        -> calculator
  OCR               -> OCR
  visual perception -> vision
  simple logic      -> logic engine
  complex reasoning -> reasoning model
  failed model      -> fallback
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class RouterDecision:
    """A logged routing decision."""

    task: str
    selected: str
    confidence: float
    reason: str
    timestamp: float = field(default_factory=time.time)
    fallback_chain: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "selected": self.selected,
            "confidence": self.confidence,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "fallback_chain": self.fallback_chain,
        }


class Router:
    """Deterministic task classification + tool selection with logged decisions."""

    TASK_PATTERNS = [
        ("arithmetic", re.compile(r"\b(compute|sum|minus|times|divide|mod|arithmetic|calculate)\b|\d\s*[+\-*/]\s*\d", re.I)),
        ("ocr", re.compile(r"\b(ocr|read\s+text|extract\s+text\s+from|transcribe\s+image)\b", re.I)),
        ("vision", re.compile(r"\b(see|look\s+at|visual|image|picture|photo|what\s+is\s+in)\b", re.I)),
        ("calculator", re.compile(r"^[0-9+\-*/()\s.]+$")),
        ("logic", re.compile(r"\b(syllogism|transitiv|if.*then|deduc|infer|logic)\b", re.I)),
        ("simple_logic", re.compile(r"\b(before|after|older|adjacent|north\s+of)\b", re.I)),
    ]

    def __init__(self, registry: Optional[Dict[str, Callable[..., Any]]] = None):
        self.registry = registry or {}
        self.log: List[RouterDecision] = []

    def register(self, name: str, handler: Callable[..., Any]) -> None:
        self.registry[name] = handler

    def classify(self, task: str) -> str:
        """Classify a task into a route name, deterministically."""
        for name, pat in self.TASK_PATTERNS:
            if pat.search(task):
                return name
        return "complex_reasoning"

    def route(self, task: str, confidence: float = 1.0) -> RouterDecision:
        """Pick a handler for the task and log the decision."""
        bucket = self._canonical_class(task)
        reason = f"classified '{task}' as '{bucket}'"
        # deterministic mapping: task -> resource
        decision = RouterDecision(task=task, selected=bucket, confidence=confidence, reason=reason)
        self.log.append(decision)
        return decision

    def _canonical_class(self, task: str) -> str:
        c = self.classify(task)
        if c == "calculator":
            return "calculator"
        return c or "complex_reasoning"

    def _needs_logic(self, task: str) -> bool:
        return bool(re.search(r"\b(if|then|proof|deriv|infer)\b", task, re.I))

    def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> Tuple[str, Any]:
        """Route and execute, with fallback if the chosen resource fails."""
        decision = self.route(task)
        ctx = context or {}
        # Primary attempt
        try:
            handler = self.registry.get(decision.selected)
            if handler is None:
                raise KeyError(f"no handler registered for '{decision.selected}'")
            result = handler(task, ctx)
            return decision.selected, result
        except Exception as e:
            # Try a fallback chain
            for name in self._fallback_chain(decision.selected):
                try:
                    h = self.registry.get(name)
                    if h is None:
                        continue
                    result = h(task, ctx)
                    decision.fallback_chain.append(name)
                    decision.reason += f" [fallback -> {name}]"
                    return name, result
                except Exception:
                    continue
            raise RuntimeError(f"all routes failed for task '{task}': {e}")

    def _fallback_chain(self, selected: str) -> List[str]:
        return {
            "arithmetic": ["calculator"],
            "calculator": ["arithmetic"],
            "vision": ["ocr"],
            "ocr": ["vision"],
            "logic": ["complex_reasoning", "simple_logic"],
            "simple_logic": ["logic", "complex_reasoning"],
        }.get(selected, [])

    def decisions(self, task: Optional[str] = None) -> List[RouterDecision]:
        if task is None:
            return list(self.log)
        return [d for d in self.log if d.task == task]

    def last_decision(self) -> Optional[RouterDecision]:
        return self.log[-1] if self.log else None

    def stats(self) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = {}
        for d in self.log:
            s = out.setdefault(d.selected, {"routes": 0, "fallbacks": 0})
            s["routes"] += 1
            if d.fallback_chain:
                s["fallbacks"] += 1
        return out