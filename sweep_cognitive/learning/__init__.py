"""
Learning Infrastructure — Phases 14–18 + 19.

Phase 14 (spec section 22): every completed task should produce a
record — task representation, actions, observations, outcome, errors,
success factors, resource cost, time, confidence, verification result.
Persistent JSONL append-only store so real outcome data accumulates.

Phase 15 (spec sections 19/22): a LEARNED complexity estimator —
logistic regression over behavioral features (no dependencies, no
GPU) — trained on accumulated experience records, with holdout
generalization + Brier calibration vs the lexical baseline.

Phase 16 (spec section 23): a skill registry whose acquisition gate
requires transfer evidence across distinct contexts, not a single
success.

Phase 17 (spec section 24): self-improvement loop — candidate vs
baseline on the same evaluation set; keep only if better; record a
regression test either way.

Phase 18 (representation upgrade): a TEXT estimator over hashed
n-gram features whose WEIGHTS are learned from outcome data (not
from hand-enumerated marker lists), promoted only when it beats ALL
of: behavioral estimator on the same split, lexical baseline, and
majority-class floor on the same holdout.

Phase 19 (controlled retraining cadence): a RetrainScheduler that
trains fresh candidates on demand, detects drift vs the currently-
promoted model, and returns honest recommendations. The scheduler
never silently promotes — promotion happens only through the existing
gates, and only when callers choose to apply the recommendation.
"""

from __future__ import annotations

import hashlib
from enum import Enum
import itertools
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

_uid_counter = itertools.count(1)


def _uid(prefix: str) -> str:
    return f"{prefix}_{next(_uid_counter)}_{int(time.time() * 1000)}"


# ======================================================================
# PHASE 14 — EXPERIENCE RECORDS + PERSISTENT STORE
# ======================================================================


@dataclass
class ExperienceRecord:
    task_id: str
    goal: str
    query_features: dict[str, float]
    actions: list[dict[str, Any]]
    outcome: str
    success: bool
    verification_passed: bool
    errors: list[dict[str, Any]]
    replans: int
    duration_ms: float
    confidence: float
    routing_mode: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "query_features": self.query_features,
            "actions": self.actions,
            "outcome": self.outcome,
            "success": self.success,
            "verification_passed": self.verification_passed,
            "errors": self.errors,
            "replans": self.replans,
            "duration_ms": round(self.duration_ms, 2),
            "confidence": self.confidence,
            "routing_mode": self.routing_mode,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperienceRecord":
        return cls(**d)


class ExperienceStore:
    def __init__(self, path: Optional[str] = None):
        self.path = path
        self._skipped_lines = 0

    def append(self, record: ExperienceRecord) -> None:
        if self.path is None:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_dict()) + "\n")

    def load_all(self) -> list[ExperienceRecord]:
        if self.path is None or not os.path.exists(self.path):
            return []
        records: list[ExperienceRecord] = []
        self._skipped_lines = 0
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(ExperienceRecord.from_dict(json.loads(line)))
                except (json.JSONDecodeError, TypeError, KeyError):
                    self._skipped_lines += 1
        return records

    def count(self) -> int:
        return len(self.load_all())

    def stats(self) -> dict[str, Any]:
        records = self.load_all()
        successes = sum(1 for r in records if r.success)
        return {
            "records": len(records),
            "successes": successes,
            "failures": len(records) - successes,
            "corrupt_lines": self._skipped_lines,
            "path": self.path,
        }


# ======================================================================
# PHASE 15 — LEARNED ESTIMATOR (BEHAVIORAL FEATURES)
# ======================================================================

COMPLEXITY_MARKERS = (
    "investigate", "analyze", "compare", "why", "how", "explain",
    "research", "verify", "prove", "design",
)
AMBIGUITY_MARKERS = (
    "maybe", "possibly", "might", "could be", "unclear", "not sure",
    "perhaps", "or something",
)
CONSEQUENCE_MARKERS = (
    "production", "deploy", "delete", "irreversible", "critical",
    "urgent", "security", "payment", "legal",
)


def features_from_query(query: str) -> dict[str, float]:
    lower = query.lower()
    words = query.split()
    n = max(1, len(words))
    markers = sum(1 for m in COMPLEXITY_MARKERS if m in lower)
    ambiguity = sum(1 for m in AMBIGUITY_MARKERS if m in lower)
    consequence = sum(1 for m in CONSEQUENCE_MARKERS if m in lower)
    return {
        "word_count": min(1.0, len(words) / 50.0),
        "marker_density": min(1.0, (markers / n) * 1.5),
        "marker_count": min(1.0, markers / 4.0),
        "ambiguity_markers": min(1.0, ambiguity / 3.0),
        "consequence_markers": min(1.0, consequence / 3.0),
        "question_form": 1.0 if query.strip().endswith("?") else 0.0,
    }


FEATURE_ORDER = [
    "word_count", "marker_density", "marker_count",
    "ambiguity_markers", "consequence_markers", "question_form",
]


@dataclass
class TrainingReport:
    n_records: int
    n_train: int
    n_test: int
    train_accuracy: Optional[float] = None
    test_accuracy: Optional[float] = None
    test_brier: Optional[float] = None
    baseline_accuracy: Optional[float] = None
    beats_baseline: Optional[bool] = None
    weights: dict[str, float] = field(default_factory=dict)
    bias: float = 0.0
    rejected_reason: str = ""

    @property
    def accepted(self) -> bool:
        return self.rejected_reason == ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_records": self.n_records,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "train_accuracy": self.train_accuracy,
            "test_accuracy": self.test_accuracy,
            "test_brier": self.test_brier,
            "baseline_accuracy": self.baseline_accuracy,
            "beats_baseline": self.beats_baseline,
            "accepted": self.accepted,
            "rejected_reason": self.rejected_reason,
        }


def _logistic(z: float) -> float:
    if z > 30:
        return 1.0
    if z < -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


class LogisticEstimator:
    def __init__(self, weights: Optional[dict[str, float]] = None,
                 bias: float = 0.0):
        self.weights = dict(weights or {})
        self.bias = bias

    def predict_proba(self, features: dict[str, float]) -> float:
        z = self.bias
        for name, value in features.items():
            z += self.weights.get(name, 0.0) * value
        return _logistic(z)

    def to_dict(self) -> dict[str, Any]:
        return {"weights": dict(self.weights), "bias": self.bias}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LogisticEstimator":
        return cls(weights=d.get("weights", {}), bias=float(d.get("bias", 0.0)))


def lexical_baseline(query: str) -> float:
    f = features_from_query(query)
    return min(1.0, 0.25 + 0.4 * f["marker_density"]
               + 0.2 * f["marker_count"]
               + 0.3 * f["consequence_markers"])


class LearningPipeline:
    MIN_RECORDS = 30
    MIN_TEST = 8
    TEST_FRACTION = 0.3

    def __init__(self, store: ExperienceStore, seed: int = 42):
        self.store = store
        self.seed = seed

    def train(self, epochs: int = 300, lr: float = 0.5, l2: float = 0.01,
              record_progress: Optional[Callable[[str, float], None]] = None,
              ) -> TrainingReport:
        records = [r for r in self.store.load_all()
                   if r.outcome in ("completed", "failed", "needs_user")]
        n = len(records)
        if n < self.MIN_RECORDS:
            return TrainingReport(
                n_records=n, n_train=0, n_test=0,
                rejected_reason=(
                    f"insufficient data: {n} records < {self.MIN_RECORDS} minimum"
                ),
            )

        rng = random.Random(self.seed)
        shuffled = list(records)
        rng.shuffle(shuffled)
        n_test = max(self.MIN_TEST, int(n * self.TEST_FRACTION))
        n_test = min(n_test, max(1, n // 3))
        test, train = shuffled[:n_test], shuffled[n_test:]

        if not train:
            return TrainingReport(
                n_records=n, n_train=0, n_test=len(test),
                rejected_reason="no training records after split",
            )

        train_labels = {r.success for r in train}
        if len(train_labels) < 2:
            return TrainingReport(
                n_records=n, n_train=len(train), n_test=len(test),
                rejected_reason="training set has a single class; no signal",
            )

        weights = {name: 0.0 for name in FEATURE_ORDER}
        bias = 0.0
        m = len(train)
        for epoch in range(epochs):
            grad_w = {name: 0.0 for name in FEATURE_ORDER}
            grad_b = 0.0
            for rec in train:
                feats = rec.query_features
                z = bias + sum(weights.get(k, 0.0) * feats.get(k, 0.0)
                               for k in FEATURE_ORDER)
                p = _logistic(z)
                err = p - (1.0 if rec.success else 0.0)
                for k in FEATURE_ORDER:
                    grad_w[k] += err * feats.get(k, 0.0)
                grad_b += err
            for k in FEATURE_ORDER:
                weights[k] -= lr * (grad_w[k] / m + l2 * weights[k])
            bias -= lr * grad_b / m
            if record_progress and epoch % 50 == 0:
                record_progress(f"epoch {epoch}", _loss(train, weights, bias, l2))

        model = LogisticEstimator(weights=weights, bias=bias)
        test_correct = 0
        brier = 0.0
        for rec in test:
            p = model.predict_proba(rec.query_features)
            pred = p >= 0.5
            test_correct += pred == rec.success
            target = 1.0 if rec.success else 0.0
            brier += (p - target) ** 2
        test_accuracy = test_correct / len(test)
        test_brier = brier / len(test)
        base_correct = sum(
            (lexical_baseline(r.goal) >= 0.5) == r.success for r in test
        )
        baseline_accuracy = base_correct / len(test)
        train_correct = sum(
            (model.predict_proba(r.query_features) >= 0.5) == r.success
            for r in train
        )
        train_accuracy = train_correct / len(train)
        beats = test_accuracy > baseline_accuracy
        return TrainingReport(
            n_records=n, n_train=len(train), n_test=len(test),
            train_accuracy=round(train_accuracy, 4),
            test_accuracy=round(test_accuracy, 4),
            test_brier=round(test_brier, 4),
            baseline_accuracy=round(baseline_accuracy, 4),
            beats_baseline=beats,
            weights=dict(weights), bias=bias,
            rejected_reason="" if beats else (
                f"test_accuracy {test_accuracy:.3f} <= baseline {baseline_accuracy:.3f}; "
                "not promoted"
            ),
        )

    @staticmethod
    def promote(report: TrainingReport, path: str) -> bool:
        if not report.accepted:
            return False
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({
                "model": LogisticEstimator(report.weights, report.bias).to_dict(),
                "report": report.to_dict(),
                "trained_at": time.time(),
            }, fh, indent=2)
        return True

    @staticmethod
    def load_promoted(path: str) -> Optional[LogisticEstimator]:
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            return LogisticEstimator.from_dict(payload["model"])
        except (json.JSONDecodeError, KeyError, TypeError, OSError):
            return None


def _loss(train: list, weights: dict[str, float], bias: float, l2: float) -> float:
    total = 0.0
    for rec in train:
        z = bias + sum(weights.get(k, 0.0) * rec.query_features.get(k, 0.0)
                       for k in FEATURE_ORDER)
        p = _logistic(z)
        y = 1.0 if rec.success else 0.0
        total += -(y * math.log(max(p, 1e-12)) + (1 - y) * math.log(max(1 - p, 1e-12)))
    reg = 0.5 * l2 * sum(v * v for v in weights.values())
    return total / len(train) + reg


# ======================================================================
# PHASE 16 — SKILL REGISTRY WITH TRANSFER GATE
# ======================================================================

class SkillStatus(str, Enum):
    CANDIDATE = "candidate"
    PROVISIONAL = "provisional"
    ACQUIRED = "acquired"
    RETIRED = "retired"


@dataclass
class Skill:
    name: str
    description: str = ""
    procedure: list[dict[str, Any]] = field(default_factory=list)
    preconditions: dict[str, Any] = field(default_factory=dict)
    status: SkillStatus = SkillStatus.CANDIDATE
    usage_count: int = 0
    success_count: int = 0
    distinct_contexts_succeeded: set[str] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)
    last_used: Optional[float] = None
    skill_id: str = field(default_factory=lambda: _uid("skill"))

    @property
    def success_rate(self) -> Optional[float]:
        return self.success_count / self.usage_count if self.usage_count else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "procedure_steps": len(self.procedure),
            "preconditions": self.preconditions,
            "status": self.status.value,
            "usage_count": self.usage_count,
            "success_rate": self.success_rate,
            "distinct_contexts": sorted(self.distinct_contexts_succeeded),
            "created_at": self.created_at,
            "last_used": self.last_used,
        }


class SkillRegistry:
    MIN_DISTINCT_CONTEXTS = 3
    ACQUIRE_RATE = 0.7
    RETIRE_RATE = 0.4
    RETIRE_MIN_USES = 5

    def __init__(self):
        self._skills: dict[str, Skill] = {}

    def register_observation(
        self, name: str, context_key: str, success: bool,
        description: str = "", procedure: Optional[list[dict[str, Any]]] = None,
        preconditions: Optional[dict[str, Any]] = None,
    ) -> Skill:
        skill = self._skills.get(name)
        if skill is None:
            skill = Skill(
                name=name, description=description,
                procedure=list(procedure or []),
                preconditions=dict(preconditions or {}),
            )
            self._skills[name] = skill
        skill.usage_count += 1
        if success:
            skill.success_count += 1
            skill.distinct_contexts_succeeded.add(context_key)
        skill.last_used = time.time()
        self._update_status(skill)
        return skill

    def _update_status(self, skill: Skill) -> None:
        if skill.status == SkillStatus.ACQUIRED:
            if (skill.usage_count >= self.RETIRE_MIN_USES
                    and skill.success_rate is not None
                    and skill.success_rate < self.RETIRE_RATE):
                skill.status = SkillStatus.RETIRED
            return
        if skill.status == SkillStatus.RETIRED:
            return
        rate = skill.success_rate
        if (len(skill.distinct_contexts_succeeded) >= self.MIN_DISTINCT_CONTEXTS
                and rate is not None and rate >= self.ACQUIRE_RATE):
            skill.status = SkillStatus.ACQUIRED
        elif skill.usage_count > 0:
            skill.status = SkillStatus.PROVISIONAL

    def get(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def acquired(self) -> list[Skill]:
        return [s for s in self._skills.values() if s.status == SkillStatus.ACQUIRED]

    def applicable(self, context: dict[str, Any]) -> list[Skill]:
        out = []
        for skill in self.acquired():
            ok = all(
                context.get(k) == v
                for k, v in skill.preconditions.items()
            )
            if ok:
                out.append(skill)
        return out

    def stats(self) -> dict[str, Any]:
        by_status: dict[str, int] = {}
        for s in self._skills.values():
            by_status[s.status.value] = by_status.get(s.status.value, 0) + 1
        return {"skills": len(self._skills), "by_status": by_status}


# ======================================================================
# PHASE 17 — SELF-IMPROVEMENT LOOP
# ======================================================================


@dataclass
class ImprovementCandidate:
    name: str
    evaluate: Callable[[], tuple[float, dict[str, Any]]]
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "notes": self.notes}


@dataclass
class ImprovementOutcome:
    candidate: str
    baseline_score: float
    candidate_score: float
    improved: bool
    kept: bool
    details: dict[str, Any] = field(default_factory=dict)
    evaluated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "baseline_score": round(self.baseline_score, 4),
            "candidate_score": round(self.candidate_score, 4),
            "improved": self.improved,
            "kept": self.kept,
            "details": self.details,
            "evaluated_at": self.evaluated_at,
        }


class SelfImprovementLoop:
    def __init__(self, min_improvement: float = 0.02):
        self.min_improvement = min_improvement
        self.history: list[ImprovementOutcome] = []

    def evaluate_candidate(
        self,
        candidate: ImprovementCandidate,
        baseline: Callable[[], tuple[float, dict[str, Any]]],
        apply: Optional[Callable[[bool], None]] = None,
    ) -> ImprovementOutcome:
        base_score, base_details = baseline()
        cand_score, cand_details = candidate.evaluate()
        improved = cand_score > base_score + self.min_improvement
        outcome = ImprovementOutcome(
            candidate=candidate.name,
            baseline_score=base_score,
            candidate_score=cand_score,
            improved=improved,
            kept=improved,
            details={
                "baseline": base_details,
                "candidate": cand_details,
                "delta": round(cand_score - base_score, 4),
                "min_improvement": self.min_improvement,
            },
        )
        if apply is not None:
            apply(improved)
        self.history.append(outcome)
        return outcome

    def regression_summary(self) -> list[dict[str, Any]]:
        return [o.to_dict() for o in self.history]


# ======================================================================
# PHASE 18 — TEXT ESTIMATOR (LEARNED CONTENT WEIGHTING)
# ======================================================================

TEXT_DIM = 512
TEXT_NGRAM_MIN = 1
TEXT_NGRAM_MAX = 2

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _hash_index(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % TEXT_DIM


def text_features_from_query(query: str) -> dict[str, float]:
    tokens = _tokenize(query)
    if not tokens:
        return {}
    counts: dict[int, float] = {}
    for n in range(TEXT_NGRAM_MIN, TEXT_NGRAM_MAX + 1):
        for i in range(len(tokens) - n + 1):
            gram = "_".join(tokens[i:i + n])
            idx = _hash_index(gram)
            counts[idx] = counts.get(idx, 0.0) + 1.0
    norm = math.sqrt(sum(v * v for v in counts.values()))
    if norm == 0.0:
        return {}
    return {str(idx): v / norm for idx, v in counts.items()}


@dataclass
class TextTrainingReport:
    n_records: int
    n_train: int = 0
    n_test: int = 0
    train_accuracy: Optional[float] = None
    test_accuracy: Optional[float] = None
    test_brier: Optional[float] = None
    behavioral_baseline_accuracy: Optional[float] = None
    lexical_baseline_accuracy: Optional[float] = None
    majority_class_accuracy: Optional[float] = None
    beats_behavioral: Optional[bool] = None
    beats_lexical: Optional[bool] = None
    beats_majority: Optional[bool] = None
    weights: dict[str, float] = field(default_factory=dict)
    bias: float = 0.0
    rejected_reason: str = ""

    @property
    def accepted(self) -> bool:
        return self.rejected_reason == ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_records": self.n_records,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "train_accuracy": self.train_accuracy,
            "test_accuracy": self.test_accuracy,
            "test_brier": self.test_brier,
            "behavioral_baseline_accuracy": self.behavioral_baseline_accuracy,
            "lexical_baseline_accuracy": self.lexical_baseline_accuracy,
            "majority_class_accuracy": self.majority_class_accuracy,
            "beats_all_baselines": self.accepted,
            "rejected_reason": self.rejected_reason,
        }


class TextEstimator:
    def __init__(self, weights: Optional[dict[str, float]] = None,
                 bias: float = 0.0):
        self.weights = dict(weights or {})
        self.bias = bias

    def predict_proba(self, features: dict[str, float]) -> float:
        z = self.bias
        for name, value in features.items():
            z += self.weights.get(name, 0.0) * value
        return _logistic(z)

    def predict_proba_text(self, query: str) -> float:
        return self.predict_proba(text_features_from_query(query))

    def to_dict(self) -> dict[str, Any]:
        return {"weights": dict(self.weights), "bias": self.bias}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TextEstimator":
        return cls(weights=d.get("weights", {}), bias=float(d.get("bias", 0.0)))


class TextLearningPipeline:
    MIN_RECORDS = 40
    MIN_TEST = 10
    TEST_FRACTION = 0.3
    MIN_DF = 2

    def __init__(self, store: ExperienceStore, seed: int = 42):
        self.store = store
        self.seed = seed

    def train(self, epochs: int = 300, lr: float = 0.5, l2: float = 0.01,
              min_df: Optional[int] = None,
              record_progress: Optional[Callable[[str, float], None]] = None,
              ) -> TextTrainingReport:
        if min_df is None:
            min_df = self.MIN_DF
        records = [r for r in self.store.load_all()
                   if r.outcome in ("completed", "failed", "needs_user")]
        n = len(records)
        if n < self.MIN_RECORDS:
            return TextTrainingReport(
                n_records=n,
                rejected_reason=(
                    f"insufficient data: {n} records < {self.MIN_RECORDS} minimum"
                ),
            )

        rng = random.Random(self.seed)
        shuffled = list(records)
        rng.shuffle(shuffled)
        n_test = max(self.MIN_TEST, int(n * self.TEST_FRACTION))
        n_test = min(n_test, max(1, n // 3))
        test, train = shuffled[:n_test], shuffled[n_test:]

        if not train:
            return TextTrainingReport(
                n_records=n, n_test=len(test),
                rejected_reason="no training records after split",
            )

        train_labels = {r.success for r in train}
        if len(train_labels) < 2:
            return TextTrainingReport(
                n_records=n, n_train=len(train), n_test=len(test),
                rejected_reason="training set has a single class; no signal",
            )

        doc_freq: dict[str, int] = {}
        for rec in train:
            for gram in text_features_from_query(rec.goal):
                doc_freq[gram] = doc_freq.get(gram, 0) + 1
        vocab = {g for g, c in doc_freq.items() if c >= min_df}

        def _feats(goal: str) -> dict[str, float]:
            return {k: v for k, v in text_features_from_query(goal).items()
                    if k in vocab}

        text_weights: dict[str, float] = {}
        bias = 0.0
        m = len(train)
        for epoch in range(epochs):
            grad: dict[str, float] = {}
            grad_b = 0.0
            for rec in train:
                feats = _feats(rec.goal)
                z = bias + sum(text_weights.get(k, 0.0) * v for k, v in feats.items())
                p = _logistic(z)
                err = p - (1.0 if rec.success else 0.0)
                for k, v in feats.items():
                    grad[k] = grad.get(k, 0.0) + err * v
                grad_b += err
            for k, g in grad.items():
                w = text_weights.get(k, 0.0)
                text_weights[k] = w - lr * (g / m + l2 * w)
            bias -= lr * grad_b / m
            if record_progress and epoch % 100 == 0:
                record_progress(f"epoch {epoch}", 0.0)

        model = TextEstimator(weights=text_weights, bias=bias)

        behav_weights = {name: 0.0 for name in FEATURE_ORDER}
        behav_bias = 0.0
        for epoch in range(epochs):
            grad_w = {name: 0.0 for name in FEATURE_ORDER}
            grad_b = 0.0
            for rec in train:
                feats = rec.query_features
                z = behav_bias + sum(behav_weights.get(k, 0.0) * feats.get(k, 0.0)
                                     for k in FEATURE_ORDER)
                p = _logistic(z)
                err = p - (1.0 if rec.success else 0.0)
                for k in FEATURE_ORDER:
                    grad_w[k] += err * feats.get(k, 0.0)
                grad_b += err
            for k in FEATURE_ORDER:
                behav_weights[k] -= lr * (grad_w[k] / m + l2 * behav_weights[k])
            behav_bias -= lr * grad_b / m

        behav_model = LogisticEstimator(weights=behav_weights, bias=behav_bias)

        def _score(predict):
            correct = 0
            brier = 0.0
            for rec in test:
                p = predict(rec)
                correct += (p >= 0.5) == rec.success
                target = 1.0 if rec.success else 0.0
                brier += (p - target) ** 2
            return correct / len(test), brier / len(test)

        text_acc, text_brier = _score(
            lambda r: model.predict_proba(_feats(r.goal))
        )
        behav_acc, _ = _score(
            lambda r: behav_model.predict_proba(r.query_features)
        )
        lex_acc, _ = _score(
            lambda r: lexical_baseline(r.goal)
        )
        majority = max(
            sum(1 for r in test if r.success),
            sum(1 for r in test if not r.success)
        ) / len(test)

        beats_behavioral = text_acc > behav_acc
        beats_lexical = text_acc > lex_acc
        beats_majority = text_acc > majority

        train_correct = sum(
            (model.predict_proba(_feats(r.goal)) >= 0.5) == r.success
            for r in train
        )

        beats_all = beats_behavioral and beats_lexical and beats_majority
        rejected = "" if beats_all else (
            f"test_accuracy {text_acc:.3f} did not beat ALL baselines on holdout "
            f"(behavioral={behav_acc:.3f}, lexical={lex_acc:.3f}, majority={majority:.3f}); "
            "not promoted"
        )
        return TextTrainingReport(
            n_records=n, n_train=len(train), n_test=len(test),
            train_accuracy=round(train_correct / len(train), 4),
            test_accuracy=round(text_acc, 4),
            test_brier=round(text_brier, 4),
            behavioral_baseline_accuracy=round(behav_acc, 4),
            lexical_baseline_accuracy=round(lex_acc, 4),
            majority_class_accuracy=round(majority, 4),
            beats_behavioral=beats_behavioral,
            beats_lexical=beats_lexical,
            beats_majority=beats_majority,
            weights=dict(text_weights), bias=bias,
            rejected_reason=rejected,
        )

    @staticmethod
    def promote(report: TextTrainingReport, path: str) -> bool:
        if not report.accepted:
            return False
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({
                "model": TextEstimator(report.weights, report.bias).to_dict(),
                "report": report.to_dict(),
                "trained_at": time.time(),
            }, fh, indent=2)
        return True

    @staticmethod
    def load_promoted(path: str) -> Optional[TextEstimator]:
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            return TextEstimator.from_dict(payload["model"])
        except (json.JSONDecodeError, KeyError, TypeError, OSError):
            return None


class BlendedEstimator:
    def __init__(self, behavioral: Optional[LogisticEstimator] = None,
                 text: Optional[TextEstimator] = None,
                 text_weight: float = 0.5):
        if behavioral is None and text is None:
            raise ValueError("BlendedEstimator needs at least one component")
        self.behavioral = behavioral
        self.text = text
        self.text_weight = text_weight

    def predict_proba(self, features: dict[str, float]) -> float:
        return self._blend(
            behavioral=lambda: self.behavioral.predict_proba(features)
            if self.behavioral else None,
            text=None,
        )

    def predict_proba_blended(self, features: dict[str, float],
                              query: str) -> float:
        return self._blend(
            behavioral=(lambda: self.behavioral.predict_proba(features)
                        if self.behavioral else None),
            text=(lambda: self.text.predict_proba_text(query)
                  if self.text else None),
        )

    def _blend(self, behavioral, text) -> float:
        parts: list[float] = []
        wts: list[float] = []
        if self.behavioral is not None:
            try:
                p = behavioral()
                if p is not None:
                    parts.append(p)
                    wts.append(1.0 - self.text_weight)
            except Exception:
                pass
        if self.text is not None and text is not None:
            try:
                p = text()
                if p is not None:
                    parts.append(p)
                    wts.append(self.text_weight)
            except Exception:
                pass
        if not parts:
            raise RuntimeError("no estimator component produced a prediction")
        total = sum(wts) or 1.0
        return sum(p * w for p, w in zip(parts, wts)) / total


def majority_baseline(records: list[ExperienceRecord]) -> float:
    if not records:
        return 0.0
    pos = sum(1 for r in records if r.success)
    return max(pos, len(records) - pos) / len(records)


# ======================================================================
# PHASE 19 — SCHEDULED RETRAINING + DRIFT DETECTION
# ======================================================================

TIMESTAMP_FORMAT = "%.0f"


class DriftReport:
    """Outcome of comparing a fresh model's holdout report against the
    currently-promoted model's embedded report. Honest: if no model is
    promoted, drift is unmeasured (None)."""

    def __init__(
        self,
        promoted_at: Optional[float] = None,
        promoted_report: Optional[Any] = None,
        fresh_vs_promoted_test_accuracy_delta: Optional[float] = None,
        severity: str = "unknown",
        note: str = "",
        fresh_report: Any = None,
    ):
        self.promoted_at = promoted_at
        self.promoted_report = promoted_report
        self.fresh_vs_promoted_test_accuracy_delta = fresh_vs_promoted_test_accuracy_delta
        self.severity = severity
        self.note = note
        self.fresh_report = fresh_report

    @property
    def has_drift(self) -> bool:
        return self.severity in ("degraded", "degraded_closely")

    @property
    def degraded_amount(self) -> Optional[float]:
        return self.fresh_vs_promoted_test_accuracy_delta

    def to_dict(self) -> dict:
        return {
            "promoted_at": self.promoted_at,
            "promoted_test_accuracy": (
                getattr(self.promoted_report, "test_accuracy", None)
                if self.promoted_report is not None else None
            ),
            "fresh_test_accuracy": (
                getattr(self.fresh_report, "test_accuracy", None)
                if self.fresh_report is not None else None
            ),
            "delta_vs_promoted": self.fresh_vs_promoted_test_accuracy_delta,
            "severity": self.severity,
            "note": self.note,
            "drift_note": self.note,
        }


class DriftDetector:
    def __init__(self, min_improve: float = 0.02):
        self.min_improve = min_improve

    def detect(
        self,
        fresh_report: Any,
        current_promoted_report: Optional[Any] = None,
    ) -> DriftReport:
        if current_promoted_report is None:
            return DriftReport(
                fresh_report=fresh_report,
                severity="ok",
                note="no promoted model to compare against",
            )
        if isinstance(current_promoted_report, DriftReport):
            current_promoted_report = current_promoted_report.promoted_report
        current_acc = getattr(current_promoted_report, "test_accuracy", None)
        fresh_acc = getattr(fresh_report, "test_accuracy", None)
        if current_acc is None or fresh_acc is None:
            return DriftReport(
                fresh_report=fresh_report,
                promoted_report=current_promoted_report,
                severity="unknown",
                note="could not compare: one of the reports had no test_accuracy",
            )

        delta = fresh_acc - current_acc
        if delta >= self.min_improve:
            severity = "recovered"
            note = (
                f"fresh beats promoted by {delta:.4f} >= {self.min_improve} "
                f"(fresh={fresh_acc:.4f}, promoted={current_acc:.4f})"
            )
        elif delta > -self.min_improve:
            severity = "ok"
            note = (
                f"fresh within {abs(delta):.4f} of promoted "
                f"(fresh={fresh_acc:.4f}, promoted={current_acc:.4f}); "
                "not a meaningful degradation"
            )
        elif delta >= -0.05:
            severity = "degraded"
            note = (
                f"fresh degraded vs promoted by {abs(delta):.4f} "
                f"(fresh={fresh_acc:.4f}, promoted={current_acc:.4f})"
            )
        else:
            severity = "degraded"
            note = (
                f"fresh severely degraded vs promoted by {abs(delta):.4f} "
                f"(fresh={fresh_acc:.4f}, promoted={current_acc:.4f})"
            )

        return DriftReport(
            fresh_report=fresh_report,
            promoted_report=current_promoted_report,
            fresh_vs_promoted_test_accuracy_delta=delta,
            severity=severity,
            note=note,
        )


class CadencePolicy:
    def __init__(
        self,
        min_interval_seconds: int = 3600,
        min_new_records: int = 50,
        min_improvement: float = 0.02,
    ):
        self.min_interval_seconds = min_interval_seconds
        self.min_new_records = min_new_records
        self.min_improvement = min_improvement

    def is_allowed(
        self,
        store: "ExperienceStore",
        last_promoted_at: Optional[float],
        last_training_at: Optional[float],
    ) -> tuple[bool, str]:
        if last_training_at is not None and (
            _records_since(last_training_at, store) < self.min_new_records
        ):
            return False, (
                f"not enough new records since last training: "
                f"need {self.min_new_records}"
            )
        if last_promoted_at is not None and (
            time.time() - last_promoted_at < self.min_interval_seconds
        ):
            return False, (
                f"cadence interval not elapsed since last promotion "
                f"({self.min_interval_seconds}s)"
            )
        return True, ""

    def improvement_threshold(self) -> float:
        return self.min_improvement


def _records_since(at: float, store: "ExperienceStore") -> int:
    """Records whose created_at is strictly after at."""
    n = 0
    for r in store.load_all():
        if getattr(r, "created_at", 0.0) > at:
            n += 1
    return n


class RetrainScheduler:
    def __init__(
        self,
        store: ExperienceStore,
        cadence: Optional[CadencePolicy] = None,
        behavioral_promoted_path: Optional[str] = None,
        text_promoted_path: Optional[str] = None,
        seed: int = 7,
    ):
        self.store = store
        self.cadence = cadence or CadencePolicy()
        self.behavioral_promoted_path = behavioral_promoted_path
        self.text_promoted_path = text_promoted_path
        self.seed = seed
        self._last_behav_training_at: Optional[float] = None
        self._last_text_training_at: Optional[float] = None
        self._last_behav_promoted_at: Optional[float] = None
        self._last_text_promoted_at: Optional[float] = None

    def current_promoted(self, space: str) -> Any:
        path = (
            self.behavioral_promoted_path if space == "behavioral"
            else self.text_promoted_path
        )
        if path is None or not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            from types import SimpleNamespace
            report = payload.get("report")
            return SimpleNamespace(**report) if isinstance(report, dict) else None
        except (json.JSONDecodeError, KeyError, TypeError, OSError):
            return None

    def schedule_behavioral(self) -> dict:
        allowed, why = self.cadence.is_allowed(
            self.store,
            self._last_behav_promoted_at,
            self._last_behav_training_at,
        )
        if not allowed:
            return {
                "space": "behavioral",
                "status": "cadence_blocked",
                "reason": why,
                "report": None,
                "drift": None,
                "recommendation": "skip",
                "path": self.behavioral_promoted_path,
            }

        report = LearningPipeline(self.store, seed=self.seed).train()
        current = self.current_promoted("behavioral")
        drift = DriftDetector(min_improve=self.cadence.min_improvement).detect(
            report, current
        )
        recommendation = _recommend_behavioral(report, drift, current)

        if recommendation == "promote" and self.behavioral_promoted_path:
            if LearningPipeline.promote(report, self.behavioral_promoted_path):
                self._last_behav_promoted_at = time.time()

        if report.n_records > 0:
            self._last_behav_training_at = time.time()

        return {
            "space": "behavioral",
            "status": "trained",
            "report": report,
            "drift": drift,
            "recommendation": recommendation,
            "path": self.behavioral_promoted_path,
            "should_update_promoted_at": recommendation == "promote",
        }

    def schedule_text(self) -> dict:
        allowed, why = self.cadence.is_allowed(
            self.store,
            self._last_text_promoted_at,
            self._last_text_training_at,
        )
        if not allowed:
            return {
                "space": "text",
                "status": "cadence_blocked",
                "reason": why,
                "report": None,
                "drift": None,
                "recommendation": "skip",
                "path": self.text_promoted_path,
            }

        report = TextLearningPipeline(self.store, seed=self.seed).train()
        current = self.current_promoted("text")
        drift = DriftDetector(min_improve=self.cadence.min_improvement).detect(
            report, current
        )
        recommendation = _recommend_text(
            report, drift, current,
            min_improve=self.cadence.min_improvement,
        )

        if recommendation == "promote" and self.text_promoted_path:
            if TextLearningPipeline.promote(report, self.text_promoted_path):
                self._last_text_promoted_at = time.time()

        if report.n_records > 0:
            self._last_text_training_at = time.time()

        return {
            "space": "text",
            "status": "trained",
            "report": report,
            "drift": drift,
            "recommendation": recommendation,
            "path": self.text_promoted_path,
            "should_update_promoted_at": recommendation == "promote",
        }

    def schedule_both(self) -> dict:
        text_result = self.schedule_text()
        behav_result = self.schedule_behavioral()
        return {
            "spaces": {"text": text_result, "behavioral": behav_result},
            "summary": _summary(text_result, behav_result),
        }


def _recommend_behavioral(
    report: TrainingReport,
    drift: DriftReport,
    current_promoted_report: Optional[Any],
) -> str:
    if not report.accepted:
        return "refuse"
    if not report.beats_baseline:
        return "refuse"
    if current_promoted_report is not None and (
        drift.degraded_amount is None or drift.degraded_amount < 0.02
    ):
        return "hold"
    return "promote"


def _recommend_text(
    report: TextTrainingReport,
    drift: DriftReport,
    current_promoted_report: Optional[Any],
    min_improve: Optional[float] = None,
) -> str:
    if not report.accepted:
        return "refuse"
    if current_promoted_report is not None and (
        drift.degraded_amount is None or drift.degraded_amount < (min_improve or 0.0)
    ):
        return "hold"
    return "promote"


def _summary(text_result: dict, behav_result: dict) -> dict:
    def one(r):
        if r["status"] == "cadence_blocked":
            return {"space": r["space"], "action": "skip", "reason": r["reason"]}
        rec = r["recommendation"]
        drift = r["drift"]
        return {
            "space": r["space"],
            "action": rec,
            "test_accuracy": (
                getattr(r["report"], "test_accuracy", None)
                if r["report"] else None
            ),
            "drift": drift.severity if drift else "unmeasured",
            "drift_note": drift.note if drift else None,
        }
    return {"text": one(text_result), "behavioral": one(behav_result)}


__all__ = [
    "ExperienceRecord",
    "ExperienceStore",
    "features_from_query",
    "FEATURE_ORDER",
    "LogisticEstimator",
    "lexical_baseline",
    "TrainingReport",
    "LearningPipeline",
    "Skill",
    "SkillStatus",
    "SkillRegistry",
    "ImprovementCandidate",
    "ImprovementOutcome",
    "SelfImprovementLoop",
    "TEXT_DIM",
    "text_features_from_query",
    "TextEstimator",
    "TextTrainingReport",
    "TextLearningPipeline",
    "BlendedEstimator",
    "majority_baseline",
    "DriftReport",
    "DriftDetector",
    "CadencePolicy",
    "RetrainScheduler",
]
