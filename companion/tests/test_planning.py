"""Tests for the Python port of intent analysis + request planning."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from companion.config import BrainSettings, get_settings
from companion.main import app
from companion.planning import (
    analyze_intent,
    is_overview_request,
    is_social,
    plan_chat_request,
)


# ── intent analysis ──────────────────────────────────────────


@pytest.mark.parametrize(
    ("message", "intent", "task"),
    [
        ("what is a CRM", "ask_question", "research"),
        ("research the latest AI trends", "request_research", "research"),
        ("summarize this doc", "request_summary", "summarization"),
        ("translate to french", "request_translation", "translation"),
        ("compare iphone vs android", "request_comparison", "reasoning"),
        ("write an email to the team", "request_writing", "writing"),
        ("hi", "chat_converse", "general"),
        ("draft a post for twitter", "request_writing", "writing"),
        ("show me my leads", "manage_data", "general"),
    ],
)
def test_analyze_intent_classifications(message: str, intent: str, task: str) -> None:
    analysis = analyze_intent(message)
    assert analysis.primary_intent == intent
    assert analysis.task_type == task


def test_analyze_intent_needs_tools() -> None:
    assert analyze_intent("research the latest AI trends").needs_tools is True
    assert analyze_intent("what is a CRM").needs_tools is False


def test_analyze_intent_suggested_actions() -> None:
    assert list(analyze_intent("research X").suggested_actions) == ["search:web", "read:url"]
    assert list(analyze_intent("summarize X").suggested_actions) == ["summarize:text"]
    assert list(analyze_intent("hi").suggested_actions) == []


def test_analyze_intent_suggested_category() -> None:
    assert analyze_intent("research X").suggested_category == "search"
    assert analyze_intent("fix the bug").suggested_category == "coding"
    assert analyze_intent("automate X").suggested_category == "automation"
    assert analyze_intent("hi").suggested_category is None


def test_analyze_intent_entities() -> None:
    analysis = analyze_intent(
        "check example.org and email test@example.com about the 2024 release"
    )
    assert "example.org" in analysis.entities
    assert "test@example.com" in analysis.entities


def test_analyze_intent_follow_up_with_history() -> None:
    history = [
        {"role": "user", "text": "what is a CRM"},
        {"role": "assistant", "text": "a CRM manages contacts"},
    ]
    analysis = analyze_intent("can you explain that more", history)
    assert analysis.is_follow_up is True


# ── overview / social detection ──────────────────────────────


def test_is_overview_request() -> None:
    assert is_overview_request("what is on my plate today") is True
    assert is_overview_request("read me today's briefing") is True
    assert is_overview_request("brief me") is True
    assert is_overview_request("give me a daily overview") is True
    assert is_overview_request("give me an overview of the project") is False
    assert is_overview_request("hi") is False


def test_is_social() -> None:
    assert is_social("hi") is True
    assert is_social("Thanks!") is True
    assert is_social("good morning") is True
    assert is_social("hey what's the weather") is False


# ── chat plan ────────────────────────────────────────────────


def test_plan_research_grounds_and_uses_research_knobs() -> None:
    plan = plan_chat_request("research the latest AI trends")
    assert plan.grounded is True
    assert plan.needs_memory is True
    assert plan.needs_knowledge is True
    assert plan.needs_overview is False
    assert plan.temperature == 0.2
    assert plan.max_tokens == 2000


def test_plan_plain_question_gets_conversational_knobs() -> None:
    plan = plan_chat_request("what is a CRM")
    assert plan.grounded is False
    assert plan.needs_memory is True
    assert plan.needs_knowledge is False
    assert plan.temperature == 0.5
    assert plan.max_tokens == 1200


def test_plan_social_pays_nothing() -> None:
    plan = plan_chat_request("hi")
    assert plan.grounded is False
    assert plan.needs_memory is False
    assert plan.needs_knowledge is False
    assert plan.needs_overview is False
    assert plan.temperature == 0.5


def test_plan_comparison_grounds() -> None:
    plan = plan_chat_request("compare iphone vs android")
    assert plan.grounded is True
    assert plan.needs_knowledge is True


def test_plan_overview_request() -> None:
    plan = plan_chat_request("what is on my plate today")
    assert plan.needs_overview is True


def test_plan_force_model() -> None:
    plan = plan_chat_request("hello there", force_model="gemini-2.5-pro")
    assert plan.model == "gemini-2.5-pro"


def test_plan_force_grounded() -> None:
    plan = plan_chat_request("hi", force_grounded=True)
    assert plan.grounded is True


def test_plan_rationale_is_human_readable() -> None:
    plan = plan_chat_request("research the latest AI trends")
    assert "intent=request_research" in plan.rationale
    assert "task=research" in plan.rationale
    assert "grounded=True" in plan.rationale


# ── HTTP API ─────────────────────────────────────────────────


def _client(settings: BrainSettings) -> TestClient:
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app, headers={"Authorization": "Bearer test-only"})


def test_plan_chat_endpoint(settings: BrainSettings) -> None:
    client = _client(settings)
    resp = client.post(
        "/api/brain/plan/chat",
        json={"user_id": "u1", "message": "research the latest AI trends"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "request_research"
    assert body["task_type"] == "research"
    assert body["grounded"] is True
    assert body["needs_memory"] is True
    assert body["temperature"] == 0.2
    assert body["protocol"] == "1"


def test_plan_chat_endpoint_forces_model(settings: BrainSettings) -> None:
    client = _client(settings)
    resp = client.post(
        "/api/brain/plan/chat",
        json={"message": "hello there", "force_model": "gemini-2.5-pro"},
    )
    assert resp.status_code == 200
    assert resp.json()["model"] == "gemini-2.5-pro"
