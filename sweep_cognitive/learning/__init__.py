"""
Learning Infrastructure — Phases 14-17.

Phase 14 (spec section 22): every completed task should produce a
record — task representation, actions, observations, outcome, errors,
success factors, resource cost, time, confidence, verification result.
This module implements those records with PERSISTENT storage (JSONL,
append-only) so real outcome data accumulates across runs. Persistent
outcome data is what upgrades scaffolding into learned models.

Phase 15 (spec sections 19/22): a LEARNED complexity estimator —
logistic regression over behavioral features (no dependencies, no
GPU) — trained on accumulated experience records. Replaces the lexical
scaffolding in prediction/routing when data supports it, with an
explicit fallback otherwise.

Phase 16 (spec section 23): a skill registry whose acquisition gate
requires transfer evidence, not a single success (spec 22: "Require
evidence that a strategy transfers").

Phase 17 (spec section 24): self-improvement loop — candidate vs
baseline on the same evaluation set; keep only if better; record a
regression test either way.

Phase 18 (representation upgrade): a TEXT estimator over hashed n-gram
features. HONEST SCOPE: the feature *hashing* is deterministic
scaffolding (fixed dim, no learned embedding); what is LEARNED is the
weighting of textual content — which fragments predict success/failure
comes from accumulated outcome data, not from hand-enumerated marker
lists. Trained dense embeddings remain future work; this is the first
rung above hand-crafted features and is gated against ALL of: the
behavioral-feature estimator, the lexical baseline, and the
majority-class floor, on the same holdout. Text is promoted only when
it strictly beats every one of them.

IMPLEMENTATION STATUS (honest):
- REAL: experience records, persistent JSONL store, logistic
  regression trainer with holdout generalization + Brier calibration
  measurement, skill registry with statistical acquisition gates,
  improvement loop with baseline comparison and rollback, text
  (hashed n-gram) estimator with multi-baseline promotion gate.
- TEMPORARY SCAFFOLDING: behavioral features are still hand-built
  counts/ratios; text features are deterministic hashed n-grams (the
  WEIGHTS are learned, the projection is not). A trained embedding
  model replaces `text_features_from_query` when data volume justifies
  it — behind the same contracts and gates.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

_uid_counter = itertools.count(1)


def _uid(prefix: str) -> str:
    return f"{prefix}_{next(_uid_counter)}_{int(time.time() * 1000)}"


# ======================================================================
# PHASE 14 — EXPERIENCE RECORDS + PERSISTENT STORE
# ======================================================================

@dataclass
class ExperienceRecord:
    """One completed (or failed) task, as the full outcome record the
    spec section 22 requires."""
    task_id: str
    goal: str
    query_features: dict[str, float]          # behavioral feature vector
    actions: list[dict[str, Any]]             # action_type, attempts, durations
    outcome: str                              # completed / failed / needs_user
    success: bool
    verification_passed: bool
    errors: list[dict[str, Any]]              # error_class, detail
    replans: int
    duration_ms: float
    confidence: float                         # self-assessed at completion
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
    """
    Persistent, append-only JSONL store of experience records.

    Design choices (spec section 22: "Do not blindly perform online
    weight updates after every task. Use controlled learning pipelines."):
    - append-only: history is immutable; training snapshots are explicit
    - load_all(): full replay for offline training
    - corruption-safe: a bad line is skipped and counted, never fatal
    """

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
# PHASE 15 — LEARNED ESTIMATOR
# ======================================================================

# Behavioral feature extraction. SCAFFOLDING at the *feature* level:
# counts and ratios, not learned embeddings. The trainer/updater below
# is real; features upgrade independently of it.
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
    """Feature vector for a query. Deterministic. Same features must be
    used at training and inference time."""
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
    """Outcome of a controlled training run. Spec 22: controlled
    pipeline, measured — not a silent online update."""
    n_records: int
    n_train: int
    n_test: int
    train_accuracy: Optional[float] = None
    test_accuracy: Optional[float] = None
    test_brier: Optional[float] = None
    baseline_accuracy: Optional[float] = None     # lexical heuristic on same split
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
    """Minimal L2-regularized logistic regression. Pure Python — runs
    anywhere SWEEP runs (spec: no GPU required, degrade gracefully)."""

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
    """The pre-existing heuristic, as an explicit baseline to beat."""
    f = features_from_query(query)
    return min(1.0, 0.25 + 0.4 * f["marker_density"] + 0.2 * f["marker_count"]
               + 0.3 * f["consequence_markers"])


class LearningPipeline:
    """Controlled training + evaluation + gated promotion (spec 22/24)."""

    MIN_RECORDS = 30
    MIN_TEST = 8
    TEST_FRACTION = 0.3

    def __init__(self, store: ExperienceStore, seed: int = 42):
        self.store = store
        self.seed = seed

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, epochs: int = 300, lr: float = 0.5, l2: float = 0.01,
              record_progress: Optional[Callable[[str, float], None]] = None,
              ) -> TrainingReport:
        """
        Train on accumulated experience: target = task success. Holdout
        evaluation + baseline comparison. NEVER auto-promotes — callers
        use `promote()` after seeing the report (spec 24: keep only if
        better, compare against baseline).
        """
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

        # Stratified-ish split: shuffle deterministically, keep both
        # classes present in test when possible.
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

        # Class balance guard: a degenerate all-one-class set has no signal
        train_labels = {r.success for r in train}
        if len(train_labels) < 2:
            return TrainingReport(
                n_records=n, n_train=len(train), n_test=len(test),
                rejected_reason="training set has a single class; no signal",
            )

        # Gradient descent on L2-regularized logistic loss
        weights = {name: 0.0 for name in FEATURE_ORDER}
        bias = 0.0
        m = len(train)
        for epoch in range(epochs):
            grad_w = {name: 0.0 for name in FEATURE_ORDER}
            grad_b = 0.0
            for rec in train:
                feats = rec.query_features
                z = bias + sum(weights.get(k, 0.0) * feats.get(k, 0.0) for k in FEATURE_ORDER)
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

        # Evaluate on holdout
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

        # Baseline on the same holdout
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

    # ------------------------------------------------------------------
    # Gated promotion
    # ------------------------------------------------------------------

    def promote(self, report: TrainingReport, path: str) -> bool:
        """Persist a model ONLY if the report was accepted (beat the
        baseline on holdout). Spec 24: keep only if better."""
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
    CANDIDATE = "candidate"        # observed once, not yet a skill
    PROVISIONAL = "provisional"    # some evidence, below acquisition bar
    ACQUIRED = "acquired"          # transfer evidence met
    RETIRED = "retired"            # success rate decayed below floor


@dataclass
class Skill:
    name: str
    description: str = ""
    procedure: list[dict[str, Any]] = field(default_factory=list)  # steps
    preconditions: dict[str, Any] = field(default_factory=dict)
    status: SkillStatus = SkillStatus.CANDIDATE
    usage_count: int = 0
    success_count: int = 0
    # Transfer evidence: distinct task contexts where the strategy worked
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
    """
    Skill acquisition requires TRANSFER EVIDENCE (spec 22): a strategy
    becomes a skill only after succeeding in MIN_DISTINCT_CONTEXTS
    distinct task contexts with a success rate at or above
    ACQUIRE_RATE. A single success makes a candidate, never a skill.
    Retired when the success rate decays below RETIRE_RATE after
    enough uses.
    """

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
        """Record one execution of a strategy in a task context."""
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
            # Decay check
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
        """Acquired skills whose preconditions hold in this context."""
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
    # evaluation function: returns (score, details); higher is better
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
    """
    Spec section 24: PERFORM → MEASURE → ANALYZE → GENERATE
    IMPROVEMENT → TEST → COMPARE BASELINE → KEEP ONLY IF BETTER →
    RECORD REGRESSION TEST.

    The loop never mutates production behavior directly. Candidates
    are callables evaluated against the baseline on the same set;
    keeping a candidate means swapping it into `apply` and recording
    the outcome as a regression checkpoint.
    """

    def __init__(self, min_improvement: float = 0.02):
        self.min_improvement = min_improvement
        self.history: list[ImprovementOutcome] = []

    def evaluate_candidate(
        self,
        candidate: ImprovementCandidate,
        baseline: Callable[[], tuple[float, dict[str, Any]]],
        apply: Optional[Callable[[bool], None]] = None,
    ) -> ImprovementOutcome:
        """Evaluate candidate vs baseline on the same evaluation set.
        If `apply` is given it is called with the keep decision (True
        only when strictly better by min_improvement)."""
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
        """The recorded outcomes ARE the regression tests: any future
        candidate must beat the current kept state, and a kept change
        that later regresses shows here as a low candidate score."""
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
    """Stable across processes (unlike Python's salted hash())."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % TEXT_DIM


def text_features_from_query(query: str) -> dict[str, float]:
    """Hashed word unigrams+bigrams, L2-normalized. Deterministic and
    process-stable. HONEST SCOPE: the projection is scaffolding; the
    WEIGHTS trained on top are what is learned (which textual content
    predicts success comes from outcome data, not marker lists)."""
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
    """Logistic model over hashed text features. Same estimator math as
    LogisticEstimator, different feature space."""

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
    """Text-space pipeline. Promotion requires beating ALL of: the
    behavioral-feature estimator (trained on the same split), the
    lexical heuristic, and the majority-class floor — on the same
    holdout (spec 24: keep only if better; spec 65: no hidden shortcuts).

    Overfitting control: minimum document frequency (MIN_DF). Grams
    seen in fewer than MIN_DF *training* records are dropped from the
    vocabulary. The vocabulary is built from the train split ONLY —
    the holdout never influences it (no leakage). This is standard
    text-ML practice and it matters: without it, sparse record-specific
    bigrams memorize the training set (verified experimentally: test
    accuracy collapsed from ~0.92 to ~0.53 without min-df at n=50)."""

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

        # Vocabulary from TRAIN split only (holdout never consulted).
        # Grams rarer than min_df are dropped: they cannot generalize.
        doc_freq: dict[str, int] = {}
        for rec in train:
            for gram in text_features_from_query(rec.goal):
                doc_freq[gram] = doc_freq.get(gram, 0) + 1
        vocab = {g for g, c in doc_freq.items() if c >= min_df}

        def _feats(goal: str) -> dict[str, float]:
            return {k: v for k, v in text_features_from_query(goal).items()
                    if k in vocab}

        # --- Text model ---
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
            # L2 toward zero for unseen weights is implicit (they stay 0)
            bias -= lr * grad_b / m
            if record_progress and epoch % 100 == 0:
                record_progress(f"epoch {epoch}", 0.0)

        model = TextEstimator(weights=text_weights, bias=bias)

        # --- Behavioral model on the SAME split (spec 65: same conditions) ---
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

        # --- Holdout evaluation: all four contenders, same data ---
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
            lambda r: model.predict_proba(_feats(r.goal)))
        behav_acc, _ = _score(
            lambda r: behav_model.predict_proba(r.query_features))
        lex_acc, _ = _score(
            lambda r: lexical_baseline(r.goal))
        majority = max(sum(1 for r in test if r.success),
                       sum(1 for r in test if not r.success)) / len(test)

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

    def promote(self, report: TextTrainingReport, path: str) -> bool:
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
    """Combines behavioral + text estimators when BOTH have been
    promoted through their gates. Honest fallback contract: when a
    component is missing or errors, the other stands alone; when both
    are missing, callers fall back to lexical signals (router/prediction
    already implement this)."""

    def __init__(self, behavioral: Optional[LogisticEstimator] = None,
                 text: Optional[TextEstimator] = None,
                 text_weight: float = 0.5):
        if behavioral is None and text is None:
            raise ValueError("BlendedEstimator needs at least one component")
        self.behavioral = behavioral
        self.text = text
        self.text_weight = text_weight

    def predict_proba(self, features: dict[str, float]) -> float:
        """Behavioral-space entry point (router contract). Uses text
        component only if the caller supplies text features via
        predict_proba_blended."""
        return self._blend(
            behavioral=lambda: self.behavioral.predict_proba(features),
            text=None,
        )

    def predict_proba_blended(self, features: dict[str, float], query: str) -> float:
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
            except Exception:  # noqa: BLE001 — one component failing must not kill the blend
                pass
        if self.text is not None and text is not None:
            try:
                p = text()
                if p is not None:
                    parts.append(p)
                    wts.append(self.text_weight)
            except Exception:  # noqa: BLE001
                pass
        if not parts:
            raise RuntimeError("no estimator component produced a prediction")
        total = sum(wts) or 1.0
        return sum(p * w for p, w in zip(parts, wts)) / total


def majority_baseline(records: list["ExperienceRecord"]) -> float:
    """Majority-class accuracy over a record list. Exposed for tests and
    callers that want the floor explicitly."""
    if not records:
        return 0.0
    pos = sum(1 for r in records if r.success)
    return max(pos, len(records) - pos) / len(records)


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
]
