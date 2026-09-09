"""Phase 14 — Feedback / Learning Loop.

Every error is classified, and a specific, targeted action is taken:
memory gets stronger, strategy adjusts, a rule is added or a model is
retrained. All feedback is written out explicitly and inspectable.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


class ErrorCategory(str, enum.Enum):
    MEMORY_GAP = "MEMORY_GAP"
    STRATEGY_ERROR = "STRATEGY_ERROR"
    RULE_GAP = "RULE_GAP"
    MODEL_BIAS = "MODEL_BIAS"
    VERIFICATION_FAILURE = "VERIFICATION_FAILURE"
    UNCLASSIFIED = "UNCLASSIFIED"


class LearningAction(str, enum.Enum):
    STRENGTHEN_MEMORY = "STRENGTHEN_MEMORY"
    ADJUST_STRATEGY = "ADJUST_STRATEGY"
    ADD_RULE = "ADD_RULE"
    RETRAIN_MODEL = "RETRAIN_MODEL"
    RAISE_UNCERTAINTY = "RAISE_UNCERTAINTY"


@dataclass
class FeedbackRecord:
    """One classified error plus the action taken."""

    feedback_id: str
    category: ErrorCategory
    error_description: str
    action: LearningAction
    target: str = ""
    resolved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feedback_id": self.feedback_id,
            "category": self.category.value,
            "error_description": self.error_description,
            "action": self.action.value,
            "target": self.target,
            "resolved": self.resolved,
        }


class FeedbackLoop:
    """Classifies errors and dispatches targeted actions via registered hooks."""

    _ACTION_BY_CATEGORY: Dict[ErrorCategory, LearningAction] = {
        ErrorCategory.MEMORY_GAP: LearningAction.STRENGTHEN_MEMORY,
        ErrorCategory.STRATEGY_ERROR: LearningAction.ADJUST_STRATEGY,
        ErrorCategory.RULE_GAP: LearningAction.ADD_RULE,
        ErrorCategory.MODEL_BIAS: LearningAction.RETRAIN_MODEL,
        ErrorCategory.VERIFICATION_FAILURE: LearningAction.RAISE_UNCERTAINTY,
        ErrorCategory.UNCLASSIFIED: LearningAction.RAISE_UNCERTAINTY,
    }

    def __init__(self):
        self.records: List[FeedbackRecord] = []
        self._handlers: Dict[LearningAction, List[Callable[[FeedbackRecord], None]]] = {}

    def register_handler(self, action: LearningAction, fn: Callable[[FeedbackRecord], None]) -> None:
        self._handlers.setdefault(action, []).append(fn)

    def classify(self, error_description: str, hints: Optional[List[str]] = None,
                 category: Optional[ErrorCategory] = None) -> ErrorCategory:
        if category is not None:
            return category
        hints = hints or []
        for hint in hints:
            h = hint.upper().replace(" ", "_")
            try:
                return ErrorCategory(h)
            except ValueError:
                continue
        return ErrorCategory.UNCLASSIFIED

    def process(self, error_description: str, hints: Optional[List[str]] = None,
                category: Optional[ErrorCategory] = None, target: str = "",
                resolved: bool = False) -> FeedbackRecord:
        cat = self.classify(error_description, hints, category)
        action = self._ACTION_BY_CATEGORY[cat]
        record = FeedbackRecord(
            feedback_id="fb{}".format(len(self.records) + 1),
            category=cat,
            error_description=error_description,
            action=action,
            target=target,
            resolved=resolved,
        )
        self.records.append(record)
        for handler in self._handlers.get(action, []):
            handler(record)
        return record

    def action_log(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self.records]

    def count_by_category(self, category: ErrorCategory) -> int:
        return len([r for r in self.records if r.category is category])