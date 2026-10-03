"""Dedupe + merge — canonical-URL dedup and cross-provider merging.

selfwatch's approach, extended: beyond merging (more providers seeing the
same URL raises confidence), Sweep adds platform/profile classification and
a 0..1 confidence score per result.
"""
from __future__ import annotations

from urllib.parse import urlparse, urlunparse

from .models import ProfileMatch, ProviderResult
from .social_parser import classify, domain as _domain

# Tracking params stripped during canonicalization
_TRACKING_PARAMS = (
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "ref", "ref_src", "igshid", "si", "s", "sk",
)


def _canonicalize(url: str) -> str:
    """Dedupe key for a URL (not for display — ``ProfileMatch.url`` keeps
    the original). Scheme is normalized to https so http/https variants of
    the same page merge."""
    p = urlparse(url)
    netloc = p.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = p.path.rstrip("/") or "/"
    # keep only non-tracking query params, sorted for stability
    kept: list[str] = []
    if p.query:
        for pair in sorted(p.query.split("&")):
            key = pair.split("=", 1)[0].lower()
            if key not in _TRACKING_PARAMS and pair:
                kept.append(pair)
    query = "&".join(kept)
    return urlunparse(("https", netloc, path, "", query, ""))


def merge(results: list[ProviderResult]) -> list[ProfileMatch]:
    """Merge provider results into deduplicated, classified matches.

    Sorting: profile-like URLs first, then by (provider agreement, domain).
    """
    by_key: dict[str, ProfileMatch] = {}
    for result in results:
        for raw in result.matches:
            key = _canonicalize(raw.url)
            platform, is_profile = classify(raw.url)
            existing = by_key.get(key)
            if existing is None:
                by_key[key] = ProfileMatch(
                    url=raw.url,
                    domain=_domain(raw.url),
                    platform=platform,
                    is_profile=is_profile,
                    title=raw.title,
                    thumbnail_url=raw.thumbnail_url,
                    sources=[result.provider],
                    score=raw.score,
                )
            else:
                if result.provider not in existing.sources:
                    existing.sources.append(result.provider)
                if not existing.title and raw.title:
                    existing.title = raw.title
                if not existing.thumbnail_url and raw.thumbnail_url:
                    existing.thumbnail_url = raw.thumbnail_url
                if existing.score is None and raw.score is not None:
                    existing.score = raw.score

    merged = list(by_key.values())
    for m in merged:
        m.confidence = _confidence(m)
    merged.sort(
        key=lambda m: (
            not m.is_profile,  # profiles first
            -len(m.sources),  # more provider agreement first
            m.domain,
        )
    )
    return merged


def _confidence(m: ProfileMatch) -> float:
    """Heuristic 0..1 confidence for a merged match.

    Signals: provider corroboration, profile-likeness, provider-native score.
    Deliberately conservative — this is a ranking prior, not a certainty.
    """
    base = 0.35
    # provider corroboration: +0.15 per extra source, capped
    base += min(0.3, 0.15 * (len(m.sources) - 1))
    if m.is_profile:
        base += 0.2
    if m.score is not None:
        try:
            s = float(m.score)
            # TinEye-style scores shrink with match distance; Lens/Yandex
            # don't provide one. Treat 0..1 scores as direct confidence.
            if 0.0 <= s <= 1.0:
                base = max(base, 0.5 + 0.4 * s)
        except (TypeError, ValueError):
            pass
    return max(0.0, min(1.0, base))
