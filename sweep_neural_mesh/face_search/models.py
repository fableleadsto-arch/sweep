"""Data models for the face / reverse-image search engine.

The engine (sweep_neural_mesh.face_search) takes a photo of a person — or any
image — and finds where it appears on the public internet, classifying results
into social profiles and general web sources.

Privacy contract (mirrors openface-search / selfwatch):
  - query images live in temp files only and are deleted after a run
  - nothing is stored between runs; there is no face database
  - only publicly reachable content is ever queried
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(slots=True)
class Face:
    """One detected face in the query image."""

    box: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    det_score: float
    embedding: Optional[list[float]] = None  # ArcFace 512-d when available
    crop_path: Optional[str] = None  # temp crop file, deleted after the run

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.box
        return max(0, x2 - x1) * max(0, y2 - y1)


@dataclass(slots=True)
class RawMatch:
    """One result as returned by a single provider, before dedupe."""

    url: str
    title: Optional[str] = None
    thumbnail_url: Optional[str] = None
    score: Optional[float] = None  # provider-native score, may be None
    source: str = ""  # provider name


@dataclass(slots=True)
class ProviderResult:
    """Outcome of one provider's search (errors are data, never exceptions)."""

    provider: str
    matches: list[RawMatch] = field(default_factory=list)
    error: Optional[str] = None
    note: Optional[str] = None
    elapsed_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(slots=True)
class ProfileMatch:
    """One deduplicated result, enriched with platform + confidence."""

    url: str
    domain: str
    platform: str = "web"  # "instagram", "linkedin", ..., "web"
    is_profile: bool = False  # True when the URL looks like a person's profile
    title: Optional[str] = None
    thumbnail_url: Optional[str] = None
    sources: list[str] = field(default_factory=list)  # providers that saw it
    score: Optional[float] = None  # provider-native score (first seen)
    confidence: float = 0.0  # 0..1, see orchestrator._confidence
    verified: Optional[bool] = None  # face-verified (None = not attempted)
    cosine: Optional[float] = None  # face cosine similarity when verified

    def render_line(self) -> str:
        """One-line human-readable summary for text reports."""
        flags = []
        if self.verified:
            flags.append("face-verified")
        elif self.verified is False:
            flags.append("face-mismatch")
        flags.append(f"{self.confidence:.2f}")
        if len(self.sources) > 1:
            flags.append(f"{len(self.sources)} sources")
        title = (self.title or "")[:60]
        suffix = f" — {title}" if title else ""
        return f"[{self.platform}] {self.url}{suffix}  ({', '.join(flags)})"

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "domain": self.domain,
            "platform": self.platform,
            "is_profile": self.is_profile,
            "title": self.title,
            "thumbnail_url": self.thumbnail_url,
            "sources": list(self.sources),
            "score": self.score,
            "confidence": round(self.confidence, 4),
            "verified": self.verified,
            "cosine": None if self.cosine is None else round(self.cosine, 4),
        }


@dataclass(slots=True)
class SearchReport:
    """Full result of one search run."""

    query_image: str  # path or URL as given by the caller
    mode: str  # "face" (person photo) | "image" (generic reverse image)
    faces_found: int = 0
    providers_used: list[str] = field(default_factory=list)
    providers_failed: dict[str, str] = field(default_factory=dict)
    results: list[ProfileMatch] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def platforms_covered(self) -> list[str]:
        seen: list[str] = []
        for r in self.results:
            if r.platform != "web" and r.platform not in seen:
                seen.append(r.platform)
        return seen

    def to_dict(self) -> dict:
        return {
            "query_image": self.query_image,
            "mode": self.mode,
            "faces_found": self.faces_found,
            "providers_used": list(self.providers_used),
            "providers_failed": dict(self.providers_failed),
            "results": [r.to_dict() for r in self.results],
            "platforms_covered": self.platforms_covered,
            "notes": list(self.notes),
            "stats": dict(self.stats),
        }
