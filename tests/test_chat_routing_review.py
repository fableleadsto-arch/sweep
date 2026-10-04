"""Regressions found while reviewing the dock's chat and result paths."""

import json

import pytest

from sweep.desktop.chat import ChatStore, result_text, understand
from sweep.parser import parse, strip_politeness


@pytest.mark.parametrize("phrase", ["hello", "hi", "thanks", "hey Sweep", "please", "can you", "thank you"])
def test_greetings_and_politeness_only_remain_conversation(phrase):
    assert understand(phrase) == ("conversation", phrase)
    assert isinstance(strip_politeness(phrase), str)


@pytest.mark.parametrize("phrase", ["talk about binary trees", "discuss the project", "chat with me", "discuss clear", "clear my downloads"])
def test_legacy_discussion_never_uses_separate_terminal_model_or_history(phrase):
    assert understand(phrase) == ("conversation", phrase)


@pytest.mark.parametrize("phrase", ["clear chat", "please reset chat", "new chat", "clear conversation", "new conversation", "reset conversation"])
def test_explicit_chat_controls_stay_in_desktop_chat(phrase):
    assert understand(phrase) == ("chat.new", phrase)


@pytest.mark.parametrize("phrase", ["find file personal-budget.csv", "search for file diary.txt", "find my note"])
def test_explicit_local_search_does_not_send_filenames_or_note_requests_to_web(phrase):
    assert understand(phrase) == ("computer.command", phrase)


def test_greeting_fix_preserves_url_payload_punctuation():
    phrase = "please open https://example.com/Case?name=Hey!"
    assert parse(phrase).params["target"] == "https://example.com/Case?name=Hey!"


def test_research_context_contains_evidence_with_attribution():
    result = {"message": "Collected one evidence item.", "evidence": [{
        "excerpt": "The documented setup requires Python 3.12.",
        "source_title": "Installation guide", "source_url": "https://example.com/install",
    }]}
    text = result_text(result)
    assert "Python 3.12" in text
    assert "Installation guide: https://example.com/install" in text


def test_evidence_output_is_bounded_and_ignores_malformed_items():
    items = [{"excerpt": "x" * 2000, "source_title": "Guide", "source_url": "https://example.com"} for _ in range(10)]
    text = result_text({"evidence": items})
    assert text.count("x" * 1200) == 8
    assert "x" * 1201 not in text
    assert len(text) <= 20000
    assert "Evidence excerpts" not in result_text({"evidence": [None, {}, {"excerpt": 1}]})


@pytest.mark.parametrize("field,value", [("title", 123), ("title", "x" * 201), ("created", 123), ("created", "not-a-date"), ("turns", [{"role": "user"}] * 201)])
def test_malformed_conversation_metadata_is_rejected_without_breaking_history(tmp_path, field, value):
    store = ChatStore(tmp_path)
    good, broken = store.create(), store.create()
    broken[field] = value
    store.path(broken["id"]).write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError):
        store.get(broken["id"])
    assert [thread["id"] for thread in store.list()] == [good["id"]]


@pytest.mark.parametrize("turn", [
    {"role": ["user"]},
    {"role": "user", "attachments": [123]},
    {"role": "user", "attachments": "file.png"},
    {"role": "user", "attachments": ["x"] * 11},
    {"role": "user", "attachments": ["x" * 1025]},
    {"role": "user", "text": "x" * 60001},
    {"role": "assistant", "task_id": 123},
    {"role": "assistant", "task_id": ""},
])
def test_malformed_conversation_turn_is_rejected(tmp_path, turn):
    store = ChatStore(tmp_path)
    thread = store.create()
    thread["turns"] = [turn]
    store.path(thread["id"]).write_text(json.dumps(thread), encoding="utf-8")
    with pytest.raises(ValueError):
        store.get(thread["id"])
    assert store.list() == []


def test_loaded_task_only_turn_and_filename_attachment_remain_compatible(tmp_path):
    store = ChatStore(tmp_path)
    thread = store.create()
    store.append(thread["id"], {"role": "user", "text": "Inspect this image", "attachments": ["image.png"]})
    store.append(thread["id"], {"role": "assistant", "task_id": "a" * 32})
    assert len(store.get(thread["id"])["turns"]) == 2


def test_append_keeps_large_history_readable_by_evicting_oldest_turns(tmp_path):
    store = ChatStore(tmp_path)
    thread = store.create()
    thread["turns"] = [{"role": "user", "text": f"{index}:" + "x" * 59980} for index in range(31)]
    store.save(thread)
    store.append(thread["id"], {"role": "user", "text": "latest:" + "x" * 59980})
    restored = store.get(thread["id"])
    assert restored["turns"][0]["text"].startswith("1:")
    assert restored["turns"][-1]["text"].startswith("latest:")
    assert store.path(thread["id"]).stat().st_size < 2_000_000
