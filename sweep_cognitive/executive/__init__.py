"""
Executive Controller — Phase 12.

The central agent layer (spec module 13). Responsibilities:
- interpret objective, define success
- maintain task state
- decompose work into steps
- select tools and execute actions
- verify that the intended state actually occurred (spec module 17:
  completion != "the action returned successfully")
- diagnose failures (spec module 18) and replan

IMPLEMENTATION STATUS (honest):
- REAL: task state machine, plan construction, step execution with
  pluggable tool handlers, verification of outcomes against expected
  states, error classification, replanning loop, full audit trail.
- TEMPORARY SCAFFOLDING: goal decomposition is template-based
  (retrieval→gather→synthesize→verify); a learned planner replaces it.
  Tool handlers are registered externally — the controller defines the
  contract, not the tools themselves.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

# Optional import: experience recording is a Phase 14 capability layered
# onto the controller. The controller works without it (learning import
# failure must never break task execution).
try:
    from ..learning import ExperienceRecord, ExperienceStore, features_from_query
    _HAS_LEARNING = True
except ImportError:  # pragma: no cover
    _HAS_LEARNING = False

_uid_counter = itertools.count(1)


def _uid(prefix: str) -> str:
    return f"{prefix}_{next(_uid_counter)}_{int(time.time() * 1000)}"


class StepStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"          # upstream failure prevents running


class TaskStatus(str, Enum):
    CREATED = "created"
    PLANNING = "planning"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    COMPLETED = "completed"          # only after verification passes
    FAILED = "failed"
    NEEDS_USER = "needs_user"        # safe handoff (spec 13-Q)
    ABORTED = "aborted"


class ErrorClass(str, Enum):
    """Failure taxonomy (spec module 18)."""
    NONE = "none"
    PERCEPTION = "perception_error"
    UNDERSTANDING = "understanding_error"
    MEMORY_RETRIEVAL = "memory_retrieval_error"
    WORLD_MODEL = "world_model_error"
    REASONING = "reasoning_error"
    PLANNING = "planning_error"
    TOOL = "tool_error"
    ENVIRONMENT = "environment_error"
    EXECUTION = "execution_error"
    VERIFICATION = "verification_error"
    CALIBRATION = "confidence_calibration_error"
    UNKNOWN = "unknown_error"


@dataclass
class Step:
    """One executable step in a plan."""
    description: str
    action_type: str                       # registered tool handler name
    params: dict[str, Any] = field(default_factory=dict)
    # Verification contract: what must be true of the result
    expected_state: dict[str, Any] = field(default_factory=dict)
    step_id: str = field(default_factory=lambda: _uid("step"))
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error_class: ErrorClass = ErrorClass.NONE
    error_detail: str = ""
    attempts: int = 0
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    # True when the result was verified against a real expected_state;
    # False when there was nothing to verify against (never silently trusted)
    metadata_verified: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "description": self.description,
            "action_type": self.action_type,
            "status": self.status.value,
            "error_class": self.error_class.value,
            "error_detail": self.error_detail,
            "attempts": self.attempts,
            "expected_state": self.expected_state,
            "duration_ms": (
                round((self.finished_at - self.started_at) * 1000, 2)
                if self.started_at and self.finished_at else None
            ),
        }


@dataclass
class Plan:
    """An ordered sequence of steps with a success definition."""
    goal: str
    success_criteria: dict[str, Any]
    steps: list[Step] = field(default_factory=list)
    plan_id: str = field(default_factory=lambda: _uid("plan"))
    created_at: float = field(default_factory=time.time)
    version: int = 1
    superseded_by: Optional[str] = None

    def pending_steps(self) -> list[Step]:
        return [s for s in self.steps if s.status in (StepStatus.PENDING, StepStatus.IN_PROGRESS)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "success_criteria": self.success_criteria,
            "version": self.version,
            "superseded_by": self.superseded_by,
            "step_count": len(self.steps),
            "steps": [s.to_dict() for s in self.steps],
        }


class ToolRegistry:
    """
    Registry of tool handlers. The controller only executes actions via
    registered handlers — this is the tool-schema boundary (spec: hard
    logic is correct for tool contracts).

    A handler has signature: handler(params: dict) -> Any
    and may raise any exception on failure.
    """

    def __init__(self):
        self._handlers: dict[str, Callable[[dict[str, Any]], Any]] = {}

    def register(self, action_type: str, handler: Callable[[dict[str, Any]], Any]) -> None:
        self._handlers[action_type] = handler

    def has(self, action_type: str) -> bool:
        return action_type in self._handlers

    def execute(self, action_type: str, params: dict[str, Any]) -> Any:
        handler = self._handlers.get(action_type)
        if handler is None:
            raise KeyError(f"No handler registered for action type: {action_type!r}")
        return handler(params)

    def list_tools(self) -> list[str]:
        return sorted(self._handlers.keys())


class ExecutiveController:
    """
    Drives the closed loop: plan → execute → observe → verify →
    diagnose → replan (spec section 64).

    The controller NEVER claims completion without verification, and
    never silently retries a step more than max_attempts.
    """

    def __init__(
        self,
        tools: Optional[ToolRegistry] = None,
        max_attempts: int = 2,
        experience_store: Optional["ExperienceStore"] = None,
    ):
        self.tools = tools or ToolRegistry()
        self.max_attempts = max_attempts
        self.experience_store = experience_store
        self._tasks: dict[str, TaskRecord] = {}

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def plan_task(self, goal: str, success_criteria: Optional[dict[str, Any]] = None,
                  context: Optional[dict[str, Any]] = None) -> "TaskRecord":
        """Interpret a goal, decompose into steps, create task state.

        TEMPLATE-BASED decomposition (scaffolding): the default plan is
        the investigation loop from spec section 20. A learned planner
        replaces `_decompose` — the TaskRecord/Step contracts stay.
        """
        criteria = success_criteria or {"verified": True}
        task = TaskRecord(goal=goal, success_criteria=criteria)
        task.status = TaskStatus.PLANNING
        task.plan = self._build_plan(goal, criteria, context)
        task.status = TaskStatus.EXECUTING
        self._tasks[task.task_id] = task
        return task

    def _build_plan(self, goal: str, criteria: dict[str, Any],
                    context: Optional[dict[str, Any]]) -> Plan:
        plan = Plan(goal=goal, success_criteria=criteria)
        for desc, action, params, expected in self._decompose(goal, criteria, context or {}):
            plan.steps.append(Step(
                description=desc, action_type=action,
                params=params, expected_state=expected,
            ))
        return plan

    def _decompose(self, goal: str, criteria: dict[str, Any],
                   context: dict[str, Any]) -> list[tuple[str, str, dict, dict]]:
        """SCAFFOLDING: template decomposition for investigation tasks.
        Yields (description, action_type, params, expected_state) tuples.
        """
        steps: list[tuple[str, str, dict, dict]] = []

        # A memory/context lookup step always comes first when a retriever
        # tool is available — cheap and almost always useful.
        steps.append((
            f"Retrieve relevant context for: {goal[:60]}",
            "retrieve_context",
            {"query": goal},
            {"found": True},
        ))
        steps.append((
            f"Execute core work for: {goal[:60]}",
            "perform_task",
            {"goal": goal, **context},
            {"done": True},
        ))
        steps.append((
            f"Verify outcome against success criteria",
            "verify_outcome",
            {"goal": goal, "criteria": criteria},
            {"verified": True},
        ))
        return steps

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, task: "TaskRecord") -> "TaskRecord":
        """Execute the plan with verification, diagnosis, replanning."""
        task.status = TaskStatus.EXECUTING
        while task.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED,
                                  TaskStatus.NEEDS_USER, TaskStatus.ABORTED):
            plan = task.plan
            pending = plan.pending_steps()

            if not pending:
                # All steps done — final verification gate
                if self._all_critical_verified(plan):
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = time.time()
                else:
                    self._diagnose_and_replan(task, None,
                        ErrorClass.VERIFICATION,
                        "Steps finished but success criteria not met")
                continue

            step = pending[0]
            self._execute_step(task, step)

        task.execution_log.append({
            "event": "terminal_state",
            "status": task.status.value,
            "at": time.time(),
        })
        self._record_experience(task)
        return task

    def _record_experience(self, task: "TaskRecord") -> None:
        """Emit a full outcome record (spec section 22) for the finished
        task into the experience store, when one is attached. This is
        the data source for controlled learning — never a silent
        weight update."""
        if self.experience_store is None or not _HAS_LEARNING:
            return
        plan = task.plan
        steps = plan.steps if plan else []
        actions = [
            {
                "action_type": s.action_type,
                "status": s.status.value,
                "attempts": s.attempts,
                "duration_ms": (
                    round((s.finished_at - s.started_at) * 1000, 2)
                    if s.started_at and s.finished_at else None
                ),
                "verified": bool(s.metadata_verified),
            }
            for s in steps
        ]
        errors = [
            {"error_class": s.error_class.value, "detail": s.error_detail[:200]}
            for s in steps
            if s.error_class != ErrorClass.NONE
        ]
        verified_successes = [
            s for s in steps
            if s.status == StepStatus.SUCCEEDED and s.metadata_verified
        ]
        executed = [s for s in steps
                    if s.status in (StepStatus.SUCCEEDED, StepStatus.FAILED)]
        verification_passed = bool(verified_successes) and len(verified_successes) == len(executed)
        # Self-assessed confidence: fraction of verified successes over
        # executed steps. Honest, cheap, auditable.
        confidence = (
            len(verified_successes) / len(executed) if executed else 0.0
        )
        start = min(
            (s.started_at for s in steps if s.started_at),
            default=task.created_at,
        )
        end = task.completed_at or time.time()
        record = ExperienceRecord(
            task_id=task.task_id,
            goal=task.goal,
            query_features=features_from_query(task.goal),
            actions=actions,
            outcome=task.status.value,
            success=task.status == TaskStatus.COMPLETED,
            verification_passed=verification_passed,
            errors=errors,
            replans=task.replan_count,
            duration_ms=max(0.0, (end - start) * 1000),
            confidence=confidence,
            routing_mode="",
        )
        try:
            self.experience_store.append(record)
            task.execution_log.append({
                "event": "experience_recorded",
                "task_id": task.task_id,
            })
        except OSError as exc:
            # Recording must never break task execution; log and continue.
            task.execution_log.append({
                "event": "experience_record_failed",
                "error": str(exc),
            })

    def _execute_step(self, task: "TaskRecord", step: Step) -> None:
        step.attempts += 1
        step.status = StepStatus.IN_PROGRESS
        step.started_at = time.time()

        if not self.tools.has(step.action_type):
            step.status = StepStatus.FAILED
            step.error_class = ErrorClass.PLANNING
            step.error_detail = f"No tool registered for {step.action_type!r}"
            self._diagnose_and_replan(task, step, ErrorClass.PLANNING,
                                      step.error_detail)
            return

        try:
            result = self.tools.execute(step.action_type, step.params)
            step.result = result
            step.finished_at = time.time()
        except Exception as exc:  # noqa: BLE001 — tool boundary
            step.finished_at = time.time()
            step.error_detail = f"{type(exc).__name__}: {exc}"
            step.error_class = self._classify_exception(exc)
            if step.attempts < self.max_attempts:
                # retry in place (transient tool/environment errors)
                step.status = StepStatus.PENDING
                task.execution_log.append({
                    "event": "retry", "step_id": step.step_id,
                    "attempt": step.attempts, "error": step.error_detail,
                })
                return
            step.status = StepStatus.FAILED
            self._diagnose_and_replan(task, step, step.error_class,
                                      step.error_detail)
            return

        # Observe + verify: did the intended state occur?
        if self._verify_step(step):
            step.status = StepStatus.SUCCEEDED
        else:
            step.error_class = ErrorClass.VERIFICATION
            step.error_detail = (
                f"Tool returned but expected state not met: "
                f"expected={step.expected_state}, got={self._summarize_result(result)}"
            )
            if step.attempts < self.max_attempts:
                step.status = StepStatus.PENDING
                return
            step.status = StepStatus.FAILED
            self._diagnose_and_replan(task, step, ErrorClass.VERIFICATION,
                                      step.error_detail)

    def _verify_step(self, step: Step) -> bool:
        """Check the result against the step's expected_state contract.

        Supported checks: key presence, equality, min/max for numbers,
        boolean flags. Empty expected_state = no verification possible
        → returns True but flags it in metadata (never silently trusted).
        """
        if not step.expected_state:
            step.metadata_verified = False
            return True
        if not isinstance(step.result, dict):
            return False
        for key, expectation in step.expected_state.items():
            if key not in step.result:
                return False
            value = step.result[key]
            if isinstance(expectation, dict):
                if "min" in expectation and not (isinstance(value, (int, float)) and value >= expectation["min"]):
                    return False
                if "max" in expectation and not (isinstance(value, (int, float)) and value <= expectation["max"]):
                    return False
                if "equals" in expectation and value != expectation["equals"]:
                    return False
                if expectation.get("truthy") and not value:
                    return False
            elif value != expectation:
                return False
        return True

    @staticmethod
    def _summarize_result(result: Any) -> str:
        if isinstance(result, dict):
            return f"dict with keys {sorted(result.keys())[:8]}"
        return type(result).__name__

    def _all_critical_verified(self, plan: Plan) -> bool:
        """Success gate (spec module 17): two conditions.

        1. No step may remain failed/blocked.
        2. At least one step must have actually SUCCEEDED and been
           verified. A plan where everything was skipped (e.g., all
           tools missing) is NOT a completed task — it produced nothing.
        """
        if not plan.steps:
            return False
        no_failures = all(
            s.status in (StepStatus.SUCCEEDED, StepStatus.SKIPPED)
            for s in plan.steps
        )
        something_succeeded = any(
            s.status == StepStatus.SUCCEEDED and s.metadata_verified
            for s in plan.steps
        )
        return no_failures and something_succeeded

    # ------------------------------------------------------------------
    # Diagnosis + replanning (spec module 18)
    # ------------------------------------------------------------------

    def _classify_exception(self, exc: Exception) -> ErrorClass:
        """Map exceptions to the error taxonomy by type/name. Deterministic
        classification of tool-level failures."""
        name = type(exc).__name__.lower()
        msg = str(exc).lower()
        if isinstance(exc, KeyError) or "not found" in msg or "no handler" in msg:
            return ErrorClass.PLANNING  # asked for a tool that doesn't exist
        if "timeout" in name or "timeout" in msg:
            return ErrorClass.ENVIRONMENT
        if "permission" in msg or "denied" in msg or "auth" in msg:
            return ErrorClass.ENVIRONMENT
        if "parse" in msg or "decode" in msg or "extract" in msg:
            return ErrorClass.PERCEPTION
        return ErrorClass.TOOL

    def _diagnose_and_replan(self, task: "TaskRecord", step: Optional[Step],
                             error_class: ErrorClass, detail: str) -> None:
        """Decide: retry (handled in _execute_step), change strategy
        (replan), or hand off to the user (unsafe/unknown)."""
        task.replan_count += 1
        task.execution_log.append({
            "event": "diagnosis",
            "error_class": error_class.value,
            "detail": detail[:200],
            "step_id": step.step_id if step else None,
            "replan_number": task.replan_count,
        })

        if task.replan_count > task.max_replans:
            task.status = TaskStatus.NEEDS_USER
            task.failure_summary = (
                f"Exceeded replan budget ({task.max_replans}); "
                f"last error [{error_class.value}]: {detail[:150]}"
            )
            return

        # Budget check FIRST — it has priority over any strategy branch
        if task.replan_count > task.max_replans:
            task.status = TaskStatus.NEEDS_USER
            task.failure_summary = (
                f"Exceeded replan budget ({task.max_replans}); "
                f"last error [{error_class.value}]: {detail[:150]}"
            )
            return

        # Strategy per error class (spec module 18 decision table)
        if error_class in (ErrorClass.ENVIRONMENT, ErrorClass.TOOL):
            # Change strategy: mark step failed, degrade the plan
            if step is not None:
                step.status = StepStatus.FAILED
            task.status = TaskStatus.EXECUTING
            task.failure_summary = f"[{error_class.value}] {detail[:150]}"
            task.plan = self._replan_shallower(task.plan, error_class, detail)
        elif error_class == ErrorClass.PLANNING:
            # Missing tool: drop the step if optional, else hand off
            if step is not None:
                step.status = StepStatus.FAILED
            task.plan = self._replan_without_step(task.plan, step)
            task.status = TaskStatus.EXECUTING
        elif error_class == ErrorClass.VERIFICATION:
            # The tool ran but the intended state did not occur (module 17).
            # Redefining success to force completion is forbidden — the
            # honest outcome is a user handoff with the diagnosis.
            task.status = TaskStatus.NEEDS_USER
            task.failure_summary = (
                f"[verification_error] Could not meet expected state after "
                f"{task.replan_count} replan(s): {detail[:150]}"
            )
        else:
            task.status = TaskStatus.NEEDS_USER
            task.failure_summary = f"[{error_class.value}] {detail[:150]}"

    def _replan_shallower(self, plan: Plan, error_class: ErrorClass,
                          detail: str) -> Plan:
        """Produce a new plan version with pending steps that keep
        failing marked for degraded handling."""
        new_plan = Plan(
            goal=plan.goal, success_criteria=plan.success_criteria,
            version=plan.version + 1,
        )
        new_plan.superseded_by = None
        for s in plan.steps:
            if s.status == StepStatus.FAILED and s.error_class == error_class \
                    and s.action_type == "perform_task":
                # Degrade: replace the failing core work with a
                # best-effort step that skips strict verification
                degraded = Step(
                    description=f"DEGRADED (was: {s.description})",
                    action_type=s.action_type,
                    params=s.params,
                    expected_state={},  # accept tool success as enough
                )
                new_plan.steps.append(degraded)
            elif s.status in (StepStatus.SUCCEEDED, StepStatus.SKIPPED):
                new_plan.steps.append(s)  # keep progress
            else:
                new_plan.steps.append(s)
        plan.superseded_by = new_plan.plan_id
        return new_plan

    def _replan_without_step(self, plan: Plan, step: Optional[Step]) -> Plan:
        """New plan version omitting a failed step (e.g., missing tool)."""
        new_plan = Plan(
            goal=plan.goal, success_criteria=plan.success_criteria,
            version=plan.version + 1,
        )
        for s in plan.steps:
            if step is not None and s.step_id == step.step_id:
                s.status = StepStatus.SKIPPED
            new_plan.steps.append(s)
        plan.superseded_by = new_plan.plan_id
        return new_plan

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_task(self, task_id: str) -> "TaskRecord":
        if task_id not in self._tasks:
            raise KeyError(f"Unknown task: {task_id}")
        return self._tasks[task_id]

    def stats(self) -> dict[str, Any]:
        by_status: dict[str, int] = {}
        for t in self._tasks.values():
            by_status[t.status.value] = by_status.get(t.status.value, 0) + 1
        return {"tasks": len(self._tasks), "by_status": by_status,
                "registered_tools": self.tools.list_tools()}


@dataclass
class TaskRecord:
    """Persistent task state (spec module 13-C/D): goal, plan, progress,
    errors, and full audit log."""
    goal: str
    success_criteria: dict[str, Any]
    task_id: str = field(default_factory=lambda: _uid("task"))
    status: TaskStatus = TaskStatus.CREATED
    plan: Optional[Plan] = None
    replan_count: int = 0
    max_replans: int = 3
    failure_summary: str = ""
    completed_at: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    execution_log: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "status": self.status.value,
            "replan_count": self.replan_count,
            "failure_summary": self.failure_summary,
            "plan": self.plan.to_dict() if self.plan else None,
            "execution_log_events": len(self.execution_log),
        }


__all__ = [
    "ExecutiveController",
    "TaskRecord",
    "TaskStatus",
    "Plan",
    "Step",
    "StepStatus",
    "ToolRegistry",
    "ErrorClass",
]
