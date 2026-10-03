"""HTML → clean Markdown extraction.

Reference architecture: crawl4ai. Instead of handing the model raw HTML we
pick the main content region, drop chrome (nav/aside/footer/script), and
emit normalized Markdown.

Uses trafilatura for robust main-content extraction, with BeautifulSoup
fallback for edge cases.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup

try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False


@dataclass
class PageMeta:
    title: str = ""
    description: str = ""
    site_name: str = ""
    author: str = ""
    published_at: Optional[str] = None
    lang: str = ""
    canonical: Optional[str] = None
    json_ld: list = field(default_factory=list)


@dataclass
class MarkdownDoc:
    url: str
    markdown: str
    text: str
    meta: PageMeta
    links: list[dict[str, str]] = field(default_factory=list)
    word_count: int = 0
    truncated: bool = False


def _decode_entities(text: str) -> str:
    """Decode common HTML entities."""
    replacements = {
        "&nbsp;": " ", "&amp;": "&", "&quot;": '"',
        "&lt;": "<", "&gt;": ">", "&mdash;": "—",
        "&ndash;": "–", "&hellip;": "…",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"&#(\d+);", lambda m: _safe_chr(int(m.group(1))), text)
    text = re.sub(r"&#x([0-9a-f]+);", lambda m: _safe_chr(int(m.group(1), 16)), text, flags=re.IGNORECASE)
    return text


def _safe_chr(code: int) -> str:
    if not (9 <= code <= 0x10FFFF):
        return " "
    try:
        return chr(code)
    except (ValueError, OverflowError):
        return " "


def _strip_noise(html: str) -> str:
    """Remove scripts, styles, noscript, SVG, iframes, forms."""
    patterns = [
        r"<!--[\s\S]*?-->",
        r"<script[\s\S]*?</script>",
        r"<style[\s\S]*?</style>",
        r"<noscript[\s\S]*?</noscript>",
        r"<svg[\s\S]*?</svg>",
        r"<iframe[\s\S]*?</iframe>",
        r"<form[\s\S]*?</form>",
    ]
    for pat in patterns:
        html = re.sub(pat, " ", html, flags=re.IGNORECASE)
    return html


def html_to_text(html: str) -> str:
    """Strip every tag and collapse whitespace — the plain-text baseline."""
    text = _strip_noise(html)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return _decode_entities(text).strip()


def _main_region(html: str) -> str:
    """Pick the densest plausible content region: article, main, or body."""
    soup = BeautifulSoup(html, "html.parser")

    candidates = []
    for tag in soup.find_all(["article", "main"]):
        candidates.append(str(tag))

    # Also look for common content containers
    for div in soup.find_all("div", attrs={"class": re.compile(r"post|content|entry|story|markdown|body", re.I)}):
        candidates.append(str(div))

    body = soup.find("body")
    body_text = html_to_text(str(body)) if body else html_to_text(html)
    candidates.append(str(body) if body else html)

    best = candidates[-1]
    best_score = 0
    for c in candidates:
        score = len(html_to_text(c))
        if score > best_score:
            best_score = score
            best = c

    # A region that lost more than 85% of body is likely a wrapper miss
    if best_score >= len(body_text) * 0.15:
        return best
    return candidates[-1]


def _extract_meta(soup: BeautifulSoup) -> PageMeta:
    """Extract page metadata from meta tags, OpenGraph, and JSON-LD."""
    meta = PageMeta()

    # Title
    if soup.title and soup.title.string:
        meta.title = _decode_entities(soup.title.string.strip())

    # Meta tags
    for tag in soup.find_all("meta"):
        name = (tag.get("name") or tag.get("property") or "").lower()
        content = tag.get("content", "")
        if name == "description":
            meta.description = content
        elif name in ("og:site_name", "site-name"):
            meta.site_name = content
        elif name == "author":
            meta.author = content
        elif name == "article:published_time":
            meta.published_at = content
        elif name == "lang":
            meta.lang = content

    # HTML lang attribute
    if not meta.lang:
        html_tag = soup.find("html")
        if html_tag:
            meta.lang = html_tag.get("lang", "")

    # Canonical
    canonical = soup.find("link", rel="canonical")
    if canonical:
        meta.canonical = canonical.get("href")

    # JSON-LD
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            if isinstance(data, list):
                meta.json_ld.extend(data)
            elif isinstance(data, dict):
                meta.json_ld.append(data)
        except (json.JSONDecodeError, TypeError):
            pass

    return meta


def _extract_links(soup: BeautifulSoup, base_url: str) -> list[dict[str, str]]:
    """Extract all links with resolved URLs."""
    links = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("#") or href.startswith("javascript:"):
            continue
        try:
            url = urljoin(base_url, href)
        except Exception:
            continue
        if url in seen:
            continue
        seen.add(url)
        text = a.get_text(strip=True)[:200]
        links.append({"url": url, "text": text})
    return links


def html_to_markdown(
    html: str,
    url: str,
    max_chars: int = 12_000,
) -> MarkdownDoc:
    """Convert HTML to clean Markdown with metadata extraction.

    Uses trafilatura for robust main-content extraction when available,
    falls back to BeautifulSoup-based extraction.
    """
    meta = _extract_meta(BeautifulSoup(html, "html.parser"))
    links = _extract_links(BeautifulSoup(html, "html.parser"), url)

    markdown = text = ""
    if HAS_TRAFILATURA:
        # Use trafilatura for robust extraction
        markdown = trafilatura.extract(
            html,
            url=url,
            include_links=True,
            include_tables=True,
            output_format="txt",
            favor_precision=False,
        ) or ""

        # Also get HTML output for link extraction
        html_out = trafilatura.extract(
            html,
            url=url,
            output_format="html",
        ) or ""

        text = html_to_text(html_out) if html_out else markdown
    if not markdown.strip() or not text.strip():
        # Fallback: BeautifulSoup-based extraction
        region = _main_region(_strip_noise(html))
        soup = BeautifulSoup(region, "html.parser")

        # Convert to rough markdown
        parts = []
        for element in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "blockquote"]):
            tag = element.name
            content = element.get_text(strip=True)
            if not content:
                continue
            if tag.startswith("h"):
                level = int(tag[1])
                parts.append(f"{'#' * level} {content}")
            elif tag == "li":
                parts.append(f"- {content}")
            elif tag == "pre":
                parts.append(f"```\n{content}\n```")
            elif tag == "blockquote":
                lines = content.split("\n")
                parts.append("\n".join(f"> {line}" for line in lines))
            else:
                parts.append(content)

        markdown = "\n\n".join(parts) or html_to_text(region)
        text = html_to_text(markdown)

    # Truncate if needed
    truncated = False
    if len(markdown) > max_chars:
        # Find a good break point
        cut = markdown.rfind(".", 0, max_chars)
        if cut < max_chars // 2:
            cut = max_chars
        markdown = markdown[:min(cut + 1, max_chars)]
        truncated = True
    if len(text) > max_chars:
        text = text[:max_chars]
        truncated = True

    word_count = len(text.split())

    return MarkdownDoc(
        url=url,
        markdown=markdown,
        text=text,
        meta=meta,
        links=links,
        word_count=word_count,
        truncated=truncated,
    )


def json_to_markdown(raw: str, url: str, max_chars: int = 12_000) -> MarkdownDoc:
    """Convert JSON content to readable Markdown."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return MarkdownDoc(url=url, markdown=raw[:max_chars], text=raw[:max_chars], meta=PageMeta())

    def _format(obj: object, depth: int = 0) -> str:
        if depth > 5:
            return str(obj)[:200]
        if isinstance(obj, dict):
            parts = []
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    parts.append(f"**{k}**:\n{_format(v, depth + 1)}")
                else:
                    parts.append(f"**{k}**: {v}")
            return "\n".join(parts)
        if isinstance(obj, list):
            return "\n".join(f"- {_format(item, depth + 1)}" for item in obj[:50])
        return str(obj)

    markdown = _format(data)
    text = html_to_text(markdown)

    return MarkdownDoc(
        url=url,
        markdown=markdown[:max_chars],
        text=text[:max_chars],
        meta=PageMeta(title="JSON Response"),
        truncated=len(markdown) > max_chars,
    )
