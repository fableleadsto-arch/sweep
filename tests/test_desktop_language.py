"""Conversational prose must not accidentally launch tools or leave the device."""
import asyncio

import pytest

from companion.providers import ProviderResult
from sweep.desktop.chat import understand
from sweep.desktop.conversation import quick_reply
from sweep.desktop.runtime import execute, route
from sweep.parser import parse


@pytest.mark.parametrize("text", [
    "show me an example", "can u show me how this works?", "show me why that is true",
    "start with the basics", "start a business", "I watched this video yesterday",
    "this image made me laugh", "I have been editing this document all morning",
    "I worked 9-5 today", "I scored 8/10 on my test", "I am 20% done with the report",
    "My budget is 20 percent of 200 dollars", "research is hard",
    "search engines are frustrating", "open source software is great",
    "open your mind", "go to bed", "look up at the stars",
    "list three ways to cheer up", "list the pros and cons", "start a conversation",
    "open up a bit", "start over", "time flies when you're having fun",
    "today was rough ngl", "date formats confuse me", "run me through your reasoning",
    "remember when we talked about my project?", "remind me how recursion works",
    "sleep is hard lately", "lock in bro", "installing this is frustrating",
    "I need to find it easier to focus", "I find it hard to focus", "find me a way to relax",
    "find the meaning of this sentence", "youtube", "calculator", "help",
    "how much ram should a laptop have?", "my system is slow", "weather is weird today",
    "yo can u explain this simply pls", "idk what to do rn", "tbh this feels overwhelming",
    "I don't want you to open calculator", "he said open youtube in brave",
    "if I say open calculator what happens?", "can you open up about this?",
])
def test_ordinary_language_stays_exactly_in_conversation(text):
    assert understand(text) == ("conversation", text)
    assert route(text) == ("conversation", text)


@pytest.mark.parametrize("text,capability", [
    ("yo can u open yt in brave pls", "computer.command"),
    ("couldya open Calculator", "computer.command"),
    ("I wanna open Character Map", "computer.command"),
    ("please show me calculator", "computer.command"),
    ("run the calculator app", "computer.command"),
    ("what time is it?", "computer.command"),
    ("what's the date?", "computer.command"),
    ("calculate 17 * 23", "computer.command"),
    ("what is 15% of 200", "computer.command"),
    ("2 plus 3", "computer.command"),
    ("find file personal-budget.csv", "computer.command"),
    ("search for file diary.txt", "computer.command"),
    ("find my note", "computer.command"),
    ("remember my favorite color is green", "computer.command"),
    ("show system info", "computer.command"),
    ("search the web for Python documentation", "web.search"),
    ("find Ada Lovelace public profiles", "web.search"),
    ("research camera lens types", "web.research"),
    ("yo can u read my screen pls", "screen.read"),
])
def test_explicit_actions_still_work_with_casual_wrappers(text, capability):
    assert understand(text)[0] == capability


def test_destination_alias_expands_without_changing_browser():
    capability, text = understand("yo can u open yt in brave pls")
    assert capability == "computer.command"
    assert parse(text).params == {"target": "youtube", "browser": "brave"}


@pytest.mark.parametrize("text", [
    "find my file medical-results.pdf", "find the file passwords.txt",
    "search my notes for bank pin", "find my note about password",
    "search for my file budget.csv", "find my documents",
    "find my password", "search my computer for receipts",
])
def test_local_lookup_terms_never_become_public_search(text):
    assert understand(text)[0] in {"computer.command", "chat.clarify"}


def test_local_note_lookup_normalizes_to_existing_query_skill():
    capability, text = understand("search my notes for Project Lumen")
    assert capability == "computer.command"
    assert parse(text).intent == "notes" and parse(text).params["action"] == "query"


def test_explicit_web_search_still_takes_the_requested_query():
    assert understand("search the web for my public project") == ("web.search", "my public project")


@pytest.mark.parametrize("target", ["https://example.com/Case?name=Hey!", r"C:\Projects\CaseSensitive.txt"])
def test_casual_command_wrappers_preserve_payload(target):
    capability, text = understand(f"couldya open {target}")
    assert capability == "computer.command"
    assert parse(text).params["target"] == target


@pytest.mark.parametrize("ending", [",", ".", "!", "?"])
def test_scrape_preserves_url_query_punctuation(ending):
    url = "https://example.com/a?x=hi" + ending
    assert understand("scrape " + url) == ("web.scrape", url)


@pytest.mark.parametrize("text", ["run echo hello", "shutdown", "install huge-package", "show me an example"])
def test_worker_cannot_force_terminal_or_ambiguous_prose_into_action(monkeypatch, text):
    monkeypatch.setattr("sweep.Controller.execute", lambda *a, **k: pytest.fail("controller reached"))
    with pytest.raises(ValueError, match="specific app"):
        asyncio.run(execute({"capability": "computer.command", "text": text, "approved": True}, lambda _: None))


@pytest.mark.parametrize("prefix", ["couldya", "wouldya", "I wanna"])
def test_approval_uses_same_normalized_action_as_execution(monkeypatch, prefix):
    monkeypatch.setattr("sweep.Controller.execute", lambda *a, **k: pytest.fail("unapproved launch"))
    with pytest.raises(PermissionError):
        asyncio.run(execute({"capability": "computer.command", "text": prefix + " open https://example.com/2"}, lambda _: None))


@pytest.mark.parametrize("text", ["yo", "sup Sweep!", "thx bro", "tysm"])
def test_short_social_replies_need_no_model(monkeypatch, text):
    monkeypatch.setattr("sweep.desktop.providers.generate_chat", lambda *a, **k: pytest.fail("model reached"))
    result = asyncio.run(execute({"capability": "conversation", "text": text, "approved": True}, lambda _: None))
    assert result["message"] == quick_reply(text)


def test_substantive_slang_uses_plain_chat_with_history_and_cannot_propose_actions(monkeypatch):
    captured, events = {}, []

    async def generate(root, **kwargs):
        captured.update(kwargs)
        kwargs["on_preview"]("That sounds exhausting.")
        return ProviderResult(text="That sounds exhausting. What happened?", parsed={
            "proposal": {"capability": "computer.command", "text": "open calculator"},
        }, provider="ollama", model="test", attempted=[])

    monkeypatch.setattr("sweep.desktop.providers.generate_chat", generate)
    text = "ngl I'm overwhelmed rn"
    result = asyncio.run(execute({"capability": "conversation", "text": text, "approved": True,
                                 "history": [{"role": "user", "content": "My project is Lumen"}]}, events.append))
    assert captured["json_mode"] is False
    assert captured["messages"] == [{"role": "user", "content": "My project is Lumen"}, {"role": "user", "content": text}]
    assert "ngl" in captured["system"] and "cannot execute tools" in captured["system"]
    assert result["message"] == "That sounds exhausting. What happened?"
    assert "proposal" not in result
    assert any(event.get("text") == "That sounds exhausting." for event in events)


def test_embedded_mesh_answers_formal_logic_without_model(monkeypatch):
    monkeypatch.setattr("sweep.desktop.providers.generate_chat", lambda *a, **k: pytest.fail("model reached"))
    text = "All cats are mammals. All mammals are animals. Are all cats animals?"
    assert understand(text)[0] == "conversation"
    result = asyncio.run(execute({"capability": "conversation", "text": text, "approved": True}, lambda _: None))
    assert result["reasoning"]["engine"] == "sweep_neural_mesh"
    assert result["reasoning"]["conclusion"] == "supported"
