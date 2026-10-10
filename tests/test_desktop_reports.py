"""Attributed local reports preserve evidence without inventing verification."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sweep.desktop import reports


@pytest.fixture
def research():
    return {
        "id": "research-session", "objective": "Research local document processing",
        "status": "complete", "started_at": "2026-10-09T05:00:00+00:00",
        "completed_at": "2026-10-09T05:01:00+00:00",
        "sources": [
            {"title": "Project documentation", "url": "https://example.org/docs",
             "retrieved_at": "2026-10-09T05:00:30+00:00", "access_mode": "public"},
            {"title": "Unavailable reference", "url": "https://example.net/reference",
             "retrieved_at": "2026-10-09T05:00:40+00:00"},
        ],
        "evidence": [{"source_url": "https://example.org/docs", "source_title": "Project documentation",
                      "excerpt": "The project reads local documents and returns plain text.",
                      "claim": "The project reads local documents.", "confidence": 0.87}],
        "actions": [{"id": "search1", "kind": "search", "description": "Searching local document processing",
                     "status": "done", "detail": "2 results"},
                    {"id": "page2", "kind": "open", "description": "Opening unavailable reference",
                     "status": "error", "detail": "Page could not be loaded"}],
    }


def test_report_writes_attributed_outputs_without_modifying_input(research, tmp_path):
    original = copy.deepcopy(research)
    result = reports.create_research_report(research, tmp_path / "task" / "artifacts")
    assert research == original
    assert result["report_summary"] == {"sources": 2, "evidence": 1, "unavailable": 1, "partial": True}
    assert [item["format"] for item in result["artifacts"]] == ["md", "json"]
    markdown = Path(result["artifacts"][0]["path"]).read_text(encoding="utf-8")
    assert "## Objective" in markdown
    assert "Research local document processing" in markdown
    assert "https://example.org/docs" in markdown
    assert "The project reads local documents" in markdown
    assert "Attributed excerpt (unverified)" in markdown
    assert "Page could not be loaded" in markdown
    assert "No excerpt was extracted from this recorded source" in markdown
    for artifact in result["artifacts"]:
        path = Path(artifact["path"])
        assert path.parent == tmp_path / "task" / "artifacts"
        assert artifact["bytes"] == path.stat().st_size
        assert artifact["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    graph = json.loads(Path(result["artifacts"][1]["path"]).read_text(encoding="utf-8"))
    assert graph == result["evidence_graph"]


def test_graph_uses_existing_cognition_contract_and_neutral_provenance(research):
    from cognition.evidence_graph import EvidenceGraph, Relation, Relationship
    from cognition.schema import object_from_dict

    graph = reports.build_research_graph(research)
    store = EvidenceGraph(in_memory=True)
    for record in graph["entities"]:
        store.add_entity(object_from_dict(record.copy()))
    for record in graph["evidence"]:
        store.add_evidence(object_from_dict(record.copy()))
    for record in graph["claims"]:
        store.add_claim(object_from_dict(record.copy()))
    for record in graph["relations"]:
        relation = Relation.from_dict(record)
        store.link(relation.claim_id, relation.evidence_id, relation.relation, relation.confidence, relation.reason)
    for record in graph["relationships"]:
        store.add_relationship(Relationship(**record))
    claim = graph["claims"][0]
    assert claim["status"] == "unverified"
    assert claim["confidence"] == 0.0
    observed = store.evidence_with_provenance(claim["id"])[0]
    assert observed["relation"] == "neutral"
    assert observed["source"] == research["sources"][0]["url"]
    assert observed["retrieval_time"] == research["sources"][0]["retrieved_at"]
    assert store.supporting_evidence(claim["id"]) == []


def test_timestamps_keep_retrieval_publication_and_unknown_meanings_separate(research):
    research["sources"][0]["published_at"] = "2025-05-15"
    research["evidence"][0]["timestamp"] = "02:31"
    graph = reports.build_research_graph(research)
    excerpt = graph["evidence"][0]
    assert excerpt["retrieval_time"] == "2026-10-09T05:00:30+00:00"
    assert excerpt["metadata"]["source_published_at"] == "2025-05-15"
    assert excerpt["metadata"]["source_timestamp_unclassified"] == "02:31"
    assert graph["sources"][1]["published_at"] is None
    report = reports.render_research_report(graph)
    assert "Publication date: 2025\\-05\\-15" in report
    assert "unknown; not inferred from retrieval" in report
    assert "Source-supplied timestamp (meaning unclassified): 02:31" in report


def test_unlisted_evidence_source_retains_citation_without_inventing_retrieval_time(research):
    research["sources"] = []
    graph = reports.build_research_graph(research)
    assert len(graph["sources"]) == 1
    assert graph["sources"][0]["url"] == research["evidence"][0]["source_url"]
    assert graph["sources"][0]["retrieved_at"] is None
    assert graph["evidence"][0]["retrieval_time"] == ""


def test_untrusted_text_cannot_embed_images_links_or_raw_html(research):
    research["objective"] = '<script>alert(1)</script> ![track](https://remote.test/pixel)'
    research["sources"][0]["title"] = '[fake](javascript:alert(1)) <img src="x">'
    research["evidence"][0]["excerpt"] = '<iframe src="https://remote.test"></iframe>\n# forged heading\n![tracking](https://remote.test/pixel)'
    markdown = reports.render_research_report(reports.build_research_graph(research))
    assert "<script>" not in markdown
    assert "<img " not in markdown
    assert "<iframe" not in markdown
    assert "![tracking]" not in markdown
    assert "\n# forged heading" not in markdown
    assert r"\!\[tracking\]" in markdown
    assert "&lt;iframe" in markdown


@pytest.mark.parametrize("url", ["javascript:alert(1)", "file:///C:/private.txt", "data:text/html,bad",
                                 "https://user:password@example.com", "https://example.com\nspoof", "https://[bad"])
def test_unsafe_source_urls_are_omitted_and_reported(research, url):
    research["sources"][0]["url"] = url
    research["evidence"][0]["source_url"] = url
    graph = reports.build_research_graph(research)
    assert graph["omitted"]["sources"] == 1
    assert graph["omitted"]["evidence"] == 1
    assert not graph["evidence"]
    assert graph["partial"] is True


def test_url_delimiters_are_encoded_in_markdown_destination(research):
    url = "https://example.org/path(test)?q=<tag>&x=1"
    research["sources"][0]["url"] = url
    research["evidence"][0]["source_url"] = url
    markdown = reports.render_research_report(reports.build_research_graph(research))
    assert "(<https://example.org/path%28test%29?q=%3Ctag%3E&x=1>)" in markdown


def test_valid_empty_research_creates_honest_partial_report(tmp_path):
    result = reports.create_research_report({"objective": "An obscure topic", "status": "complete"}, tmp_path / "artifacts")
    assert result["report_summary"] == {"sources": 0, "evidence": 0, "unavailable": 0, "partial": True}
    markdown = Path(result["artifacts"][0]["path"]).read_text(encoding="utf-8")
    assert "No usable evidence excerpts were collected" in markdown
    assert "cannot establish findings" in markdown


@pytest.mark.parametrize("session", [None, [], {}, {"objective": " "}, {"objective": 15},
                                     {"objective": "test", "sources": "invalid"}])
def test_invalid_research_does_not_create_artifact_directory(tmp_path, session):
    with pytest.raises(ValueError):
        reports.create_research_report(session, tmp_path / "artifacts")
    assert not (tmp_path / "artifacts").exists()


def test_failed_and_quarantined_actions_do_not_discard_successful_evidence(research):
    research["status"] = "failed"
    research["error"] = "Time budget reached"
    research["actions"].append({"kind": "extract", "description": "Quarantined: page", "status": "running"})
    graph = reports.build_research_graph(research)
    assert graph["partial"] is True
    assert len(graph["evidence"]) == 1
    assert len(graph["unavailable_actions"]) == 2
    report = reports.render_research_report(graph)
    assert "Time budget reached" in report
    assert "Quarantined: page" in report


def test_bounded_records_and_text_are_disclosed(research, monkeypatch):
    monkeypatch.setattr(reports, "MAX_SOURCES", 1)
    monkeypatch.setattr(reports, "MAX_EVIDENCE", 1)
    monkeypatch.setattr(reports, "MAX_ACTIONS", 1)
    research["evidence"][0]["excerpt"] = "x" * 5_000
    research["evidence"].append(copy.deepcopy(research["evidence"][0]))
    graph = reports.build_research_graph(research)
    assert graph["omitted"] == {"sources": 1, "evidence": 1, "actions": 1}
    assert graph["text_truncated"] is True
    assert len(graph["evidence"][0]["content"]) == 4_000
    assert graph["partial"] is True


def test_duplicate_records_have_stable_identity_and_do_not_inflate_counts(research):
    first = reports.build_research_graph(research)
    research["sources"].append(copy.deepcopy(research["sources"][0]))
    research["evidence"].append(copy.deepcopy(research["evidence"][0]))
    second = reports.build_research_graph(research)
    assert second["entities"] == first["entities"]
    assert second["evidence"] == first["evidence"]
    assert second["claims"] == first["claims"]


def test_multiple_excerpts_for_one_claim_share_claim_identity(research):
    another = copy.deepcopy(research["evidence"][0])
    another["excerpt"] = "Additional source detail about the document processing behavior."
    research["evidence"].append(another)
    graph = reports.build_research_graph(research)
    assert len(graph["claims"]) == 1
    assert len(graph["claims"][0]["evidence_ids"]) == 2
    assert len(graph["relations"]) == 2


@pytest.mark.parametrize("score", [float("nan"), float("inf"), 10 ** 1000, True, "not a score"])
def test_unusable_scores_are_not_serialized_as_numeric_confidence(research, score):
    research["evidence"][0]["confidence"] = score
    graph = reports.build_research_graph(research)
    assert graph["evidence"][0]["metadata"]["engine_score"] is None
    json.dumps(graph, allow_nan=False)


def test_non_string_claim_is_ignored_without_losing_excerpt(research):
    research["evidence"][0]["claim"] = 123
    graph = reports.build_research_graph(research)
    assert not graph["claims"]
    assert len(graph["evidence"]) == 1


def test_output_budget_is_checked_before_any_files_are_written(research, tmp_path, monkeypatch):
    monkeypatch.setattr(reports, "MAX_OUTPUT_BYTES", 100)
    with pytest.raises(ValueError, match="output limit"):
        reports.create_research_report(research, tmp_path / "artifacts")
    assert not (tmp_path / "artifacts").exists()


def test_repeated_export_never_overwrites_previous_artifacts(research, tmp_path):
    first = reports.create_research_report(research, tmp_path / "artifacts")
    second = reports.create_research_report(research, tmp_path / "artifacts")
    paths = {artifact["path"] for result in (first, second) for artifact in result["artifacts"]}
    assert len(paths) == 4
    assert all(Path(path).is_file() for path in paths)


def test_network_artifact_directory_is_rejected(research):
    with pytest.raises(ValueError, match="local task storage"):
        reports.create_research_report(research, r"\\server\share\artifacts")


def test_redirected_artifact_directory_is_rejected(research, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    linked = tmp_path / "artifacts"
    try:
        linked.symlink_to(elsewhere, target_is_directory=True)
    except OSError:
        pytest.skip("Creating symlinks is not permitted on this machine")
    with pytest.raises(ValueError, match="redirected"):
        reports.create_research_report(research, linked)
    assert not list(elsewhere.iterdir())


def test_failed_second_write_removes_only_new_partial_artifacts(research, tmp_path, monkeypatch):
    directory = tmp_path / "artifacts"
    directory.mkdir()
    previous = directory / "previous.md"
    previous.write_text("Keep this", encoding="utf-8")
    original = Path.open

    def broken(path, *args, **kwargs):
        if path.suffix == ".json" and args and args[0] == "xb":
            raise OSError("Simulated full disk")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", broken)
    with pytest.raises(OSError, match="full disk"):
        reports.create_research_report(research, directory)
    assert list(directory.iterdir()) == [previous]
    assert previous.read_text(encoding="utf-8") == "Keep this"
