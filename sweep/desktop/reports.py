"""Local, attributed reports over actual research observations.

This adapter reuses the cognition schema without starting a reasoner, loading a
model or fetching anything. Extracted page claims remain unverified. The caller
must derive artifact_dir from its own task storage, never from user input.
"""
from __future__ import annotations

import hashlib
import html
import json
import math
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote, urlsplit

from cognition.schema import Claim, Entity, Evidence, object_to_dict

MAX_SOURCES = 64
MAX_EVIDENCE = 256
MAX_ACTIONS = 512
MAX_OUTPUT_BYTES = 3_000_000


def _text(value, limit=4_000) -> str:
    if not isinstance(value, str):
        return ""
    return "".join(character for character in value[:limit] if ord(character) >= 32 or character in "\n\t").strip()


def _md(value: str) -> str:
    # External text cannot become raw HTML, an image, a Markdown link or a heading.
    escaped = html.escape(value, quote=False)
    return re.sub(r"([\\`*_{}\[\]()#+.!|>~-])", r"\\\1", escaped)


def _url(value) -> str:
    if not isinstance(value, str) or len(value) > 2_048 or any(character.isspace() or ord(character) < 32 for character in value):
        return ""
    try:
        parsed = urlsplit(value)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            return ""
        _ = parsed.port
    except ValueError:
        return ""
    return value


def _link(title: str, url: str) -> str:
    return f"[{_md(title)}](<{quote(url, safe=':/?&=%+#@!$,;~-._')}>)"


def _identifier(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:24]}"


def _truncated(record: dict, key: str, limit: int) -> bool:
    return isinstance(record.get(key), str) and len(record[key]) > limit


def _records(value: dict, key: str, limit: int) -> tuple[list[dict], int]:
    records = value.get(key, [])
    if not isinstance(records, list):
        raise ValueError(f"Research {key} must be a list of records.")
    selected = records[:limit]
    return [record for record in selected if isinstance(record, dict)], len(records) - sum(isinstance(record, dict) for record in selected)


def build_research_graph(session: dict) -> dict:
    """Normalize bounded observations into cognition-compatible graph objects.

    Source times retain their original labels. A generic evidence ``timestamp``
    is not assumed to be publication or retrieval time. Neutral relations record
    provenance; they do not assert that the extracted statement is true.
    """
    if not isinstance(session, dict) or not _text(session.get("objective"), 8_000):
        raise ValueError("A research result with a non-empty objective is required.")
    objective = _text(session["objective"], 8_000)
    raw_sources, skipped_sources = _records(session, "sources", MAX_SOURCES)
    raw_evidence, skipped_evidence = _records(session, "evidence", MAX_EVIDENCE)
    raw_actions, skipped_actions = _records(session, "actions", MAX_ACTIONS)
    generated_at = datetime.now(UTC).isoformat()
    objective_id = _identifier("ent", "research_objective", objective)
    objective_node = Entity(id=objective_id, name=objective, kind="research_objective", created_at=0, updated_at=0)
    sources = {}
    for record in raw_sources:
        url = _url(record.get("url"))
        if not url:
            skipped_sources += 1
            continue
        if url in sources:
            continue
        sources[url] = {
            "id": _identifier("ent", "web_source", url), "url": url,
            "title": _text(record.get("title"), 500) or url,
            "retrieved_at": _text(record.get("retrieved_at"), 100) or None,
            "published_at": _text(record.get("published_at"), 100) or None,
            "access_mode": _text(record.get("access_mode"), 50) or "unknown",
        }
    evidence, claims, relations = [], [], []
    seen = set()
    truncated = len(session["objective"]) > 8_000
    for record in raw_evidence:
        url = _url(record.get("source_url"))
        excerpt = _text(record.get("excerpt"))
        if not url or not excerpt:
            skipped_evidence += 1
            continue
        if url not in sources:
            if len(sources) >= MAX_SOURCES:
                skipped_evidence += 1
                continue
            sources[url] = {"id": _identifier("ent", "web_source", url), "url": url,
                            "title": _text(record.get("source_title"), 500) or url,
                            "retrieved_at": None, "published_at": None,
                            "access_mode": _text(record.get("access_mode"), 50) or "unknown"}
        key = (url, excerpt, _text(record.get("claim"), 1_000))
        if key in seen:
            continue
        seen.add(key)
        source = sources[url]
        truncated = truncated or _truncated(record, "excerpt", 4_000) or _truncated(record, "claim", 1_000)
        score = record.get("confidence")
        try:
            score = float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) else None
            score = score if score is not None and math.isfinite(score) else None
        except OverflowError:
            score = None
        metadata = {"source_id": source["id"], "source_title": source["title"],
                    "source_published_at": source["published_at"],
                    "source_timestamp_unclassified": _text(record.get("timestamp"), 100) or None,
                    "engine_score": score, "verification": "unverified",
                    "provenance": "Excerpt collected by app.research; no independent factual verification."}
        item = Evidence(id=_identifier("ev", *key), content=excerpt, source=url,
                        retrieval_time=source["retrieved_at"] or "", entity_ids=[objective_id, source["id"]],
                        content_hash=hashlib.sha256(excerpt.encode("utf-8")).hexdigest(),
                        metadata=metadata, created_at=0, updated_at=0)
        evidence.append(object_to_dict(item))
        if key[2]:
            claim = Claim(id=_identifier("clm", url, key[2]), statement=key[2],
                          status="unverified", confidence=0.0, evidence_ids=[item.id],
                          entity_ids=[objective_id, source["id"]], created_at=0, updated_at=0)
            existing = next((candidate for candidate in claims if candidate["id"] == claim.id), None)
            if existing:
                existing["evidence_ids"].append(item.id)
            else:
                claims.append(object_to_dict(claim))
            relations.append({"claim_id": claim.id, "evidence_id": item.id, "relation": "neutral",
                              "confidence": 0.0, "reason": "Attributed excerpt; truth of the statement has not been assessed."})
    actions = [{key: _text(record.get(key), 1_000 if key in {"description", "detail"} else 100)
                for key in ("id", "kind", "description", "status", "detail", "started_at", "ended_at")}
               for record in raw_actions]
    unavailable = [action for action in actions if action["status"] in {"error", "failed", "cancelled"}
                   or "quarantined" in action["description"].casefold()]
    omitted = {"sources": skipped_sources, "evidence": skipped_evidence, "actions": skipped_actions}
    limitations = [
        "This bounded collection is not an exhaustive search of the web.",
        "Excerpts are attributed source statements, not verified facts or an independently generated synthesis.",
        "Engine scores are retrieval heuristics, not factual confidence; claim/relation confidence 0 means not assessed.",
        "Source retrieval records can include attempted pages with no extracted evidence. An unknown publication date stays unknown.",
        "Sources may disagree. No automatic cross-source contradiction or identity determination was performed.",
        "The evidence graph is a portable data artifact; an interactive graph viewer is not included.",
    ]
    if any(omitted.values()) or truncated:
        limitations.append("Some records or text were omitted by report size/type/URL limits; the original task result is retained separately.")
    status = _text(session.get("status"), 100) or "unknown"
    error = _text(session.get("error"), 1_000)
    partial = bool(unavailable or error or status != "complete" or not evidence or any(omitted.values()) or truncated)
    entities = [object_to_dict(objective_node)] + [object_to_dict(Entity(
        id=source["id"], name=source["title"], kind="web_source", attributes=source, created_at=0, updated_at=0,
    )) for source in sources.values()]
    relationships = [{"id": _identifier("rel", objective_id, source["id"]), "subject_id": objective_id,
                      "predicate": "collected_source", "object_id": source["id"], "source": source["url"],
                      "provenance": "Research source record; not an assertion of successful page access.", "metadata": {}}
                     for source in sources.values()]
    return {"schema": "sweep.research.evidence_graph", "schema_version": "1.0.0", "generated_at": generated_at,
            "objective": objective, "session_id": _text(session.get("id"), 100), "session_status": status,
            "started_at": _text(session.get("started_at"), 100) or None,
            "completed_at": _text(session.get("completed_at"), 100) or None,
            "entities": entities, "evidence": evidence, "claims": claims, "relations": relations,
            "relationships": relationships, "sources": list(sources.values()), "actions": actions,
            "unavailable_actions": unavailable, "error": error or None, "partial": partial,
            "omitted": omitted, "text_truncated": truncated, "limitations": limitations}


def render_research_report(graph: dict) -> str:
    lines = ["# Sweep research report", "", "## Objective", "", _md(graph["objective"]), "",
             "## Collection status", "", f"- Research status: {_md(graph['session_status'])}",
             f"- Report: {'partial collection' if graph['partial'] else 'bounded collection'}",
             f"- Sources recorded: {len(graph['sources'])}", f"- Attributed excerpts: {len(graph['evidence'])}",
             f"- Unavailable or quarantined steps: {len(graph['unavailable_actions'])}",
             f"- Started: {_md(graph['started_at'] or 'unknown')}",
             f"- Completed: {_md(graph['completed_at'] or 'not recorded')}",
             f"- Report generated: {_md(graph['generated_at'])}", "",
             "Generation time is not a source publication or retrieval date."]
    if graph["error"]:
        lines.extend(["", "Research error: " + _md(graph["error"])])
    lines.extend(["", "## Sources and attributed excerpts", ""])
    if not graph["sources"]:
        lines.append("No usable public source URLs were collected.")
    if not graph["evidence"]:
        lines.append("No usable evidence excerpts were collected. This report cannot establish findings.")
    for index, source in enumerate(graph["sources"], 1):
        lines.extend(["", f"### {index}. {_link(source['title'], source['url'])}", "",
                      f"- Source URL: {_link(source['url'], source['url'])}",
                      f"- Retrieval record: {_md(source['retrieved_at'] or 'unknown')}",
                      f"- Publication date: {_md(source['published_at'] or 'unknown; not inferred from retrieval')}",
                      f"- Access mode: {_md(source['access_mode'])}", ""])
        items = [item for item in graph["evidence"] if item["source"] == source["url"]]
        if not items:
            lines.append("No excerpt was extracted from this recorded source.")
        for item in items:
            lines.extend(["Attributed excerpt (unverified):", "",
                          "\n".join("> " + _md(line) for line in item["content"].splitlines()), ""])
            timestamp = item["metadata"]["source_timestamp_unclassified"]
            if timestamp:
                lines.extend(["Source-supplied timestamp (meaning unclassified): " + _md(timestamp), ""])
    lines.extend(["", "## Research activity", ""])
    if not graph["actions"]:
        lines.append("No activity records were supplied.")
    for action in graph["actions"]:
        lines.append(f"- {_md(action['status'] or 'unknown')}: {_md(action['description'] or action['kind'] or 'unnamed step')}"
                     + (" — " + _md(action["detail"]) if action["detail"] else ""))
    lines.extend(["", "## Limits and unresolved work", ""])
    lines.extend("- " + _md(limit) for limit in graph["limitations"])
    if any(graph["omitted"].values()):
        lines.append("- Omitted records: " + ", ".join(f"{key}={count}" for key, count in graph["omitted"].items()))
    return "\n".join(lines).strip() + "\n"


def create_research_report(session: dict, artifact_dir: str | Path) -> dict:
    """Save a Markdown report and JSON graph inside a trusted task directory."""
    graph = build_research_graph(session)
    documents = [("research-report", "md", render_research_report(graph).encode("utf-8")),
                 ("research-evidence", "json", json.dumps(graph, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8"))]
    if any(len(content) > MAX_OUTPUT_BYTES for _, _, content in documents):
        raise ValueError("The research report exceeds its local output limit.")
    directory = Path(os.path.abspath(artifact_dir))
    if str(directory).startswith(("\\\\", "//")) or directory.resolve() != directory or directory.is_symlink() or directory.is_junction():
        raise ValueError("Research artifacts require local task storage without redirected directories.")
    directory.mkdir(parents=True, exist_ok=True)
    if directory.resolve() != directory:
        raise ValueError("The research artifact directory changed while preparing the report.")
    artifacts, created = [], []
    try:
        token = uuid.uuid4().hex[:12]
        for label, suffix, content in documents:
            target = directory / f"{label}-{token}.{suffix}"
            with target.open("xb") as stream:
                created.append(target)
                stream.write(content)
            artifacts.append({"name": target.name, "path": str(target), "bytes": len(content), "format": suffix,
                              "sha256": hashlib.sha256(content).hexdigest()})
    except OSError:
        for target in created:
            target.unlink(missing_ok=True)
        raise
    summary = {"sources": len(graph["sources"]), "evidence": len(graph["evidence"]),
               "unavailable": len(graph["unavailable_actions"]), "partial": graph["partial"]}
    return {"artifacts": artifacts, "evidence_graph": graph, "report_summary": summary,
            "report_message": "Saved an attributed research report and evidence graph. "
                              + ("The report preserves partial results and unresolved work." if graph["partial"]
                                 else "Source statements remain unverified.")}
