"""Page extraction — turns raw fetched content into rich PageData.

Everything flows through the Markdown layer first: clean text, semantic
link hints, headings, metadata, structured data and injection assessment.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

from ..core.types import (
    HeadingData, InjectionAssessment, LinkData, PageData, SearchAccessMode,
)
from ..core.guard import assess_injection
from ..scraping.markdown import html_to_markdown, json_to_markdown, MarkdownDoc


# ── Link Intent Detection ─────────────────────────────────────────────

LINK_INTENTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bpricing\b", re.I), "pricing"),
    (re.compile(r"\bdocs?\b|documentation|api\s*reference|developers?\b", re.I), "documentation"),
    (re.compile(r"\bfaq\b|frequently\s+asked", re.I), "faq"),
    (re.compile(r"\babout\b|our\s+story|who\s+we\s+are", re.I), "about"),
    (re.compile(r"\bcontact\b|reach\s+us|support\b|help\b", re.I), "contact"),
    (re.compile(r"\bgithub\.com\b|source\s+code|open\s+source", re.I), "github"),
    (re.compile(r"\bblog\b|news\b|articles?\b", re.I), "blog"),
    (re.compile(r"\bchangelog\b|release\s+notes\b|what'?s\s+new\b", re.I), "changelog"),
    (re.compile(r"\blogin\b|sign\s*in\b", re.I), "login"),
    (re.compile(r"\bsign\s*up\b|register|get\s+started|try\s+free", re.I), "signup"),
    (re.compile(r"\bfeatures?\b|capabilities\b", re.I), "features"),
    (re.compile(r"\bterms\b|privacy\b", re.I), "legal"),
]


def _link_intent(text: str, url: str) -> Optional[str]:
    haystack = f"{text} {url}"[:120]
    for pattern, intent in LINK_INTENTS:
        if pattern.search(haystack):
            return intent
    return None


# ── Headings ──────────────────────────────────────────────────────────

def _headings_from_markdown(markdown: str) -> list[HeadingData]:
    headings: list[HeadingData] = []
    for m in re.finditer(r"^(#{1,6})\s+(.*)$", markdown, re.MULTILINE):
        level = len(m.group(1))
        text = re.sub(r"[`*_()\[\]]", "", m.group(2)).strip()
        if not text:
            continue
        heading_id = re.sub(r"[^a-z0-9\s-]", "", text.lower()).strip().replace(" ", "-")
        headings.append(HeadingData(level=level, text=text, id=heading_id))
        if len(headings) >= 80:
            break
    return headings


# ── Main Extraction ───────────────────────────────────────────────────

def extract_page_data(
    html: Optional[str] = None,
    json_str: Optional[str] = None,
    url: str = "",
    status: int = 200,
    content_type: str = "text/html",
    max_chars: int = 12_000,
    fetched_at: Optional[str] = None,
) -> PageData:
    """Convert raw content (HTML or JSON) into PageData."""
    max_chars = min(max(max_chars, 500), 60_000)

    is_json = bool(json_str) or (html and re.match(r"^\s*(?:\[|\{)", html[:200]))

    if json_str or is_json:
        doc = json_to_markdown(json_str or html or "", url, max_chars)
    else:
        doc = html_to_markdown(html or "", url, max_chars)

    # Build links with intent detection
    links = []
    base_hostname = ""
    try:
        base_hostname = urlparse(url).hostname or ""
    except Exception:
        pass

    for link in doc.links:
        intent = _link_intent(link["text"], link["url"])
        external = False
        try:
            external = urlparse(link["url"]).hostname != base_hostname
        except Exception:
            pass
        links.append(LinkData(url=link["url"], text=link["text"], intent=intent, external=external))

    headings = _headings_from_markdown(doc.markdown)
    injection = assess_injection(doc.text)

    # Metadata
    metadata = {}
    if doc.meta.description:
        metadata["description"] = doc.meta.description
    if doc.meta.site_name:
        metadata["siteName"] = doc.meta.site_name
    if doc.meta.author:
        metadata["author"] = doc.meta.author
    if doc.meta.published_at:
        metadata["publishedAt"] = doc.meta.published_at
    if doc.meta.canonical:
        metadata["canonical"] = doc.meta.canonical
    if doc.meta.lang:
        metadata["lang"] = doc.meta.lang

    return PageData(
        url=url,
        title=doc.meta.title,
        description=doc.meta.description or None,
        text=doc.text,
        markdown=doc.markdown,
        links=links,
        headings=headings,
        metadata=metadata,
        structured_data=doc.meta.json_ld if doc.meta.json_ld else None,
        truncated=doc.truncated,
        fetched_at=fetched_at or datetime.now(timezone.utc).isoformat(),
        status=status,
        content_type=content_type,
        access_mode=SearchAccessMode.PUBLIC,
        injection=injection,
    )


# ── Chunking ──────────────────────────────────────────────────────────

def chunk_text(text: str, max_chars: int = 1800) -> list[str]:
    """Split cleaned text into ~1800-char chunks at sentence boundaries."""
    chunks: list[str] = []
    rest = text.strip()
    while rest:
        if len(rest) <= max_chars:
            chunks.append(rest)
            break
        slice_ = rest[:max_chars]
        # Find a good boundary
        candidates = [slice_.rfind("\n\n"), slice_.rfind("."), slice_.rfind("?")]
        cutoff = 1200 if len(slice_) >= 1200 else 600
        boundary = max((b for b in candidates if b > cutoff), default=-1)
        if boundary < cutoff:
            boundary = max_chars
        chunks.append(slice_[:boundary].strip())
        rest = rest[boundary:].strip()
        if len(chunks) >= 40:
            break
    return chunks


def extract_around_match(
    text: str,
    query: str,
    window_chars: int = 1600,
) -> list[dict]:
    """Extract chunk(s) of text around a keyword match."""
    needle = query.lower()
    sections = []
    lines = text.split("\n")
    current_heading = None

    for i, line in enumerate(lines):
        if re.match(r"^#{1,6}\s", line):
            current_heading = re.sub(r"^#{1,6}\s+", "", line).strip()
        if needle not in line.lower():
            continue

        target = max(0, i - window_chars // 160)
        start_line = target
        end_line = min(len(lines), target + window_chars // 120)

        section_text = "\n".join(lines[start_line:end_line]).strip()
        if section_text:
            sections.append({
                "heading": current_heading,
                "section": section_text,
                "start": start_line,
                "end": end_line,
            })
            if len(sections) >= 5:
                break

    return sections


def chunk_budget(max_tokens: int) -> int:
    """How many chunks a token budget allows."""
    return max(1, max_tokens // 8)
