"""
Tests for the executive controller (Phase 12).

Verifies:
1. Planning decomposes goals into executable steps
2. Execution runs real registered tools in order
3. Verification gate: completion requires expected states, not just
   "tool returned successfully"
4. Error diagnosis classifies failures into the taxonomy
5. Replanning: degraded plans, step omission, replan budget, user handoff
6. Missing tools fail honestly (planning error, not fake success)
7. Audit trail integrity
"""

import pytest
import time

from sweep_cognitive.executive import (
    ExecutiveController,
    TaskRecord,
    TaskStatus,
    StepStatus,
    Step,
    Plan,
    ToolRegistry,
    ErrorClass,
)


class FlakyTool:
    """A tool that fails N times before succeeding (tests retries)."""
    def __init__(self, failures: int = 1):
        self.failures = failures
        self.calls = 0

    def __call__(self, params):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("transient network timeout")
        return {"done": True, "value": 42}


def _good_tools():
    tools = ToolRegistry()
    tools.register("retrieve_context", lambda p: {"found": True, "items": []})
    tools.register("perform_task", lambda p: {"done": True, "output": f"worked on {p.get('goal', '')[:20]}"})
    tools.register("verify_outcome", lambda p: {"verified": True})
    return tools


class TestPlanning:
    def test_plan_creates_steps(self):
        ctrl = ExecutiveController(tools=_good_tools())
        task = ctrl.plan_task("investigate the bug", success_criteria={"verified": True})
        assert task.status == TaskStatus.EXECUTING
        assert task.plan is not None
        assert len(task.plan.steps) == 3  # template: retrieve, perform, verify
        assert task.plan.steps[0].action_type == "retrieve_context"

    def test_plan_ids_unique(self):
        ctrl = ExecutiveController()
        t1 = ctrl.plan_task("a")
        t2 = ctrl.plan_task("b")
        assert t1.task_id != t2.task_id

    def test_default_success_criteria(self):
        ctrl = ExecutiveController()
        task = ctrl.plan_task("do a thing")
        assert task.success_criteria == {"verified": True}


class TestExecution:
    def test_happy_path_completes(self):
        ctrl = ExecutiveController(tools=_good_tools())
        task = ctrl.plan_task("find the answer")
        result = ctrl.run(task)
        assert result.status == TaskStatus.COMPLETED
        assert result.completed_at is not None
        assert all(s.status == StepStatus.SUCCEEDED for s in result.plan.steps)

    def test_steps_run_in_order(self):
        order = []
        tools = ToolRegistry()
        tools.register("retrieve_context", lambda p: order.append("retrieve") or {"found": True})
        tools.register("perform_task", lambda p: order.append("perform") or {"done": True})
        tools.register("verify_outcome", lambda p: order.append("verify") or {"verified": True})
        ctrl = ExecutiveController(tools=tools)
        task = ctrl.plan_task("x")
        ctrl.run(task)
        assert order == ["retrieve", "perform", "verify"]

    def test_retry_on_transient_failure(self):
        flaky = FlakyTool(failures=1)
        tools = _good_tools()
        tools.register("perform_task", flaky)
        ctrl = ExecutiveController(tools=tools, max_attempts=2)
        task = ctrl.plan_task("x")
        result = ctrl.run(task)
        assert result.status == TaskStatus.COMPLETED
        assert flaky.calls == 2  # failed once, retried, succeeded

    def test_execution_log_records_retry(self):
        tools = _good_tools()
        tools.register("perform_task", FlakyTool(failures=1))
        ctrl = ExecutiveController(tools=tools, max_attempts=2)
        task = ctrl.plan_task("x")
        ctrl.run(task)
        retries = [e for e in task.execution_log if e["event"] == "retry"]
        assert len(retries) == 1


class TestVerificationGate:
    def test_unverified_tool_result_does_not_complete(self):
        """Spec module 17: completion != 'the action returned successfully'."""
        tools = ToolRegistry()
        # Tool "succeeds" but never produces the expected keys
        tools.register("retrieve_context", lambda p: {"found": False})
        tools.register("perform_task", lambda p: {"done": False})
        tools.register("verify_outcome", lambda p: {"verified": False})
        ctrl = ExecutiveController(tools=tools, max_attempts=1)
        task = ctrl.plan_task("x")
        result = ctrl.run(task)
        assert result.status != TaskStatus.COMPLETED

    def test_numeric_expectations(self):
        ctrl = ExecutiveController(tools=_good_tools())
        task = ctrl.plan_task("x", success_criteria={"count": {"min": 1}})
        plan = task.plan
        plan.steps[1].expected_state = {"value": {"min": 10}}
        tools = ToolRegistry()
        tools.register("retrieve_context", lambda p: {"found": True})
        tools.register("perform_task", lambda p: {"value": 5})  # below min
        tools.register("verify_outcome", lambda p: {"verified": True})
        ctrl.tools = tools
        result = ctrl.run(task)
        assert result.status != TaskStatus.COMPLETED

    def test_numeric_expectations_pass(self):
        ctrl = ExecutiveController(tools=_good_tools())
        task = ctrl.plan_task("x")
        tools = ToolRegistry()
        tools.register("retrieve_context", lambda p: {"found": True})
        tools.register("perform_task", lambda p: {"value": 50})
        tools.register("verify_outcome", lambda p: {"verified": True})
        ctrl.tools = tools
        task.plan.steps[1].expected_state = {"value": {"min": 10, "max": 100}}
        result = ctrl.run(task)
        assert result.status == TaskStatus.COMPLETED

    def test_no_expectations_flagged_not_trusted(self):
        """Empty expected_state can't verify anything; controller notes it."""
        step = Step(description="d", action_type="t", params={}, expected_state={})
        ctrl = ExecutiveController()
        assert ctrl._verify_step(step) is True
        assert step.metadata_verified is False


class TestDiagnosis:
    def test_missing_tool_is_planning_error(self):
        ctrl = ExecutiveController(tools=ToolRegistry())  # no tools at all
        task = ctrl.plan_task("x", success_criteria={})
        # Remove retrieve_context from the plan template by clearing tools
        result = ctrl.run(task)
        assert result.status != TaskStatus.COMPLETED
        planning_fails = [s for s in result.plan.steps
                          if s.error_class == ErrorClass.PLANNING]
        assert planning_fails  # the first missing tool is diagnosed

    def test_transient_failure_classified_environment(self):
        tools = _good_tools()
        def boom(p):
            raise TimeoutError("connection timeout after 30s")
        tools.register("perform_task", boom)
        ctrl = ExecutiveController(tools=tools, max_attempts=1)
        task = ctrl.plan_task("x")
        result = ctrl.run(task)
        assert result.status != TaskStatus.COMPLETED
        env_fails = [s for s in result.plan.steps
                     if s.error_class == ErrorClass.ENVIRONMENT]
        assert env_fails

    def test_replan_budget_leads_to_user_handoff(self):
        """After exhausting replans, safe handoff (spec 13-Q), not silent fail."""
        tools = ToolRegistry()
        def always_timeout(p):
            raise TimeoutError("environment down")
        tools.register("retrieve_context", always_timeout)
        tools.register("perform_task", always_timeout)
        tools.register("verify_outcome", always_timeout)
        ctrl = ExecutiveController(tools=tools, max_attempts=1)
        task = ctrl.plan_task("x")
        task.max_replans = 2
        result = ctrl.run(task)
        assert result.status == TaskStatus.NEEDS_USER
        assert "Exceeded replan budget" in result.failure_summary

    def test_replan_creates_new_plan_version(self):
        tools = _good_tools()
        calls = {"n": 0}
        def failing_core(p):
            calls["n"] += 1
            raise RuntimeError("permission denied")
        tools.register("perform_task", failing_core)
        ctrl = ExecutiveController(tools=tools, max_attempts=1)
        task = ctrl.plan_task("x")
        task.max_replans = 3
        result = ctrl.run(task)
        assert result.plan.version >= 2  # replanning produced a new version
        diagnoses = [e for e in result.execution_log if e["event"] == "diagnosis"]
        assert len(diagnoses) >= 1


class TestAuditTrail:
    def test_terminal_state_logged(self):
        ctrl = ExecutiveController(tools=_good_tools())
        task = ctrl.plan_task("finish cleanly")
        ctrl.run(task)
        events = [e["event"] for e in task.execution_log]
        assert "terminal_state" in events
        assert task.execution_log[-1]["status"] == "completed"

    def test_to_dict_safe(self):
        ctrl = ExecutiveController(tools=_good_tools())
        task = ctrl.plan_task("serialize me")
        ctrl.run(task)
        d = task.to_dict()
        assert d["status"] == "completed"
        assert d["plan"]["step_count"] == 3

    def test_stats(self):
        ctrl = ExecutiveController(tools=_good_tools())
        ctrl.plan_task("one")
        ctrl.plan_task("two")
        stats = ctrl.stats()
        assert stats["tasks"] == 2
        assert "retrieve_context" in stats["registered_tools"]


class TestScaffoldingHonesty:
    def test_template_decomposition_marked(self):
        import sweep_cognitive.executive as mod
        doc = mod.__doc__ or ""
        assert "TEMPORARY SCAFFOLDING" in doc


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
