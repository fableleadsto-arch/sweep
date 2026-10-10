"""End-to-end file routing, permissions, and task-owned output regressions."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from sweep.desktop.chat import understand
from sweep.desktop.runtime import CAPABILITIES, execute, run_worker


@pytest.mark.parametrize("name,text,capability", [
    ("data.csv", "summarize this data", "data.inspect"),
    ("data.json", "convert to CSV", "data.transform"),
    ("data.tsv", "filter where age >= 18", "data.transform"),
    ("data.db", "inspect table 'contacts'", "data.inspect"),
    ("notes.pdf", "summarize this document", "documents.inspect"),
    ("notes.docx", "read this document", "documents.inspect"),
    ("notes.md", "what does this say?", "documents.inspect"),
])
def test_attachments_route_to_registered_local_tools(name, text, capability):
    assert understand(text, [name]) == (capability, text)
    assert CAPABILITIES[capability].execution == "local"


def test_unsupported_data_operations_clarify_without_starting_model():
    capability, message = understand("join these datasets", ["data.csv"])
    assert capability == "chat.clarify"
    assert "not available" in message


@pytest.mark.parametrize("text", ["summarize this PDF", "clean this dataset", "read that document"])
def test_missing_document_never_fakes_a_result(text):
    assert understand(text)[0] == "chat.clarify"


def test_inspect_authority_cannot_create_data_output(tmp_path):
    source = tmp_path / "data.csv"
    source.write_text("age\n10\n20\n", encoding="utf-8")
    request = {"capability": "data.inspect", "text": "filter where age >= 18",
               "approved": True, "granted_paths": [str(source)], "_artifact_dir": str(tmp_path / "output")}
    with pytest.raises(PermissionError, match="separate"):
        asyncio.run(execute(request, lambda event: None))
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("capability", ["documents.inspect", "data.inspect", "data.transform"])
def test_file_tasks_require_dispatch_approval(capability):
    with pytest.raises(PermissionError):
        asyncio.run(execute({"capability": capability, "text": "inspect"}, lambda event: None))


def test_transform_worker_uses_task_workspace_and_preserves_source(tmp_path):
    source = tmp_path / "data.csv"
    content = b"name,age\nAda,30\nBob,12\n"
    source.write_bytes(content)
    workspace = tmp_path / "task"
    workspace.mkdir()
    task = workspace / "task.json"
    nominated = tmp_path / "not-authorized"
    task.write_text(json.dumps({"capability": "data.transform", "text": "filter where age >= 18 and export to JSON",
        "granted_paths": [str(source)], "approved": True, "_artifact_dir": str(nominated)}), encoding="utf-8")
    events = workspace / "events.jsonl"
    assert run_worker(str(task), str(events)) == 0
    result = json.loads(events.read_text(encoding="utf-8").splitlines()[-1])["data"]
    artifact = result["artifacts"][0]
    from pathlib import Path
    output = Path(artifact["path"])
    assert output.parent == workspace / "artifacts"
    assert json.loads(output.read_text(encoding="utf-8")) == [{"name": "Ada", "age": "30"}]
    assert source.read_bytes() == content
    assert not nominated.exists()


def test_document_question_uses_bounded_local_excerpt(tmp_path, monkeypatch):
    from sweep.desktop import providers
    source = tmp_path / "notes.txt"
    source.write_text("The project is called Sweep.\n" + "supporting text " * 1000, encoding="utf-8")
    sent = []
    async def generate(root, **kwargs):
        sent.append(kwargs)
        return SimpleNamespace(text="The project is Sweep.")
    monkeypatch.setattr(providers, "generate_chat", generate)
    request = {"capability": "documents.inspect", "text": "What is the project called?",
               "approved": True, "granted_paths": [str(source)]}
    result = asyncio.run(execute(request, lambda event: None))
    assert result["summary"] == "The project is Sweep."
    assert "first 5,000" in result["summary_scope"]
    assert len(sent[0]["messages"][0]["content"]) < 6000
    assert "untrusted" in sent[0]["system"]
    assert len(result["text"]) > 5000


def test_document_extract_survives_unavailable_local_model(tmp_path, monkeypatch):
    from sweep.desktop import providers
    source = tmp_path / "notes.txt"
    source.write_text("Keep this readable text.", encoding="utf-8")
    async def unavailable(*args, **kwargs):
        raise RuntimeError("The local model is unavailable.")
    monkeypatch.setattr(providers, "generate_chat", unavailable)
    result = asyncio.run(execute({"capability": "documents.inspect", "text": "summarize this",
        "approved": True, "granted_paths": [str(source)]}, lambda event: None))
    assert result["text"] == "Keep this readable text."
    assert "could not finish" in result["message"]


def test_research_worker_exports_cited_report_and_evidence(tmp_path, monkeypatch):
    from app.research import engine
    session_data = {'objective': 'Sweep test', 'status': 'completed', 'sources': [
        {'title': 'Test guide', 'url': 'https://example.com/guide', 'retrieved_at': '2026-10-09T12:00:00+00:00'}],
        'evidence': [{'id': 'e1', 'source_url': 'https://example.com/guide', 'excerpt': 'The test guide describes local tools.'}],
        'actions': []}
    session = SimpleNamespace(status=SimpleNamespace(value='completed'), error=None,
        evidence=session_data['evidence'], sources=session_data['sources'], model_dump=lambda **kwargs: session_data.copy())
    async def start(*args, **kwargs):
        return session
    monkeypatch.setattr(engine, 'start_research', start)
    task = tmp_path / 'task.json'
    task.write_text(json.dumps({'capability': 'web.research', 'text': 'Sweep test', 'approved': True}), encoding='utf-8')
    events = tmp_path / 'events.jsonl'
    assert run_worker(str(task), str(events)) == 0
    result = json.loads(events.read_text(encoding='utf-8').splitlines()[-1])['data']
    from pathlib import Path
    report = Path(result['artifacts'][0]['path'])
    assert report.parent == tmp_path / 'artifacts'
    assert 'https://example.com/guide' in report.read_text(encoding='utf-8')
    assert result['report_summary']['sources'] == 1
    assert result['evidence_graph']['evidence']
