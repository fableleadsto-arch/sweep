"""
Perception Engine — Phase 2 precursor (started in Phase 1).

The perception engine converts raw input into structured representations
with confidence and provenance. Each modality has its own front-end,
but all produce compatible shared representations.

Key principle: Do NOT force every modality into an identical raw representation.
Use modality-specific front ends followed by compatible shared representations.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from ..representation import (
    Representation,
    TextRepresentation,
    MultimodalRepresentation,
    EmbeddingVector,
    RepresentationQuality,
    RepresentationType,
)
from .document import (
    DocumentPerceptionProcessor,
    StructuredDataPerceptionProcessor,
)


class Modality(str, Enum):
    """Supported perception modalities."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"
    WEB_PAGE = "web_page"
    SCREEN = "screen"
    STRUCTURED = "structured"
    SYSTEM_STATE = "system_state"


@dataclass
class PerceptionConfidence:
    """Confidence metadata for a perception result."""
    overall: float = 0.5
    modality_confidence: dict[str, float] = field(default_factory=dict)
    quality_markers: list[str] = field(default_factory=list)
    uncertainty_markers: list[str] = field(default_factory=list)
    provenance: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall,
            "modality_confidence": self.modality_confidence,
            "quality_markers": self.quality_markers,
            "uncertainty_markers": self.uncertainty_markers,
            "provenance": self.provenance,
        }


@dataclass
class PerceptionResult:
    """
    Result of perceiving some input.

    Contains:
    - The structured representation
    - Confidence information
    - What modalities were detected
    - What was extracted
    - What's missing/uncertain
    """
    representation: Representation
    confidence: PerceptionConfidence
    detected_modalities: list[Modality] = field(default_factory=list)
    extracted_features: dict[str, Any] = field(default_factory=dict)
    missing_information: list[str] = field(default_factory=list)
    processing_latency_ms: float = 0.0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "representation": self.representation.to_dict(),
            "confidence": self.confidence.to_dict(),
            "detected_modalities": [m.value for m in self.detected_modalities],
            "extracted_features": self.extracted_features,
            "missing_information": self.missing_information,
            "processing_latency_ms": self.processing_latency_ms,
            "created_at": self.created_at,
        }


class PerceptionEngine:
    """
    Multimodal perception engine.

    Routes input to appropriate modality-specific processors,
    then produces unified representations.

    Current implementation includes:
    - Text perception (full)
    - Document perception (structure detection, format handling)
    - Structured data perception (JSON, CSV, tables)
    - Image/audio/video: metadata-only fallback
    """

    def __init__(self):
        self._text_processor = TextPerceptionProcessor()
        self._document_processor = DocumentPerceptionProcessor()
        self._structured_processor = StructuredDataPerceptionProcessor()
        # Placeholder processors for other modalities
        # These will be implemented in Phase 2
        self._image_processor = None
        self._audio_processor = None
        self._video_processor = None

    def perceive(self, input_data: Any, modality: Modality | str | None = None,
                **context) -> PerceptionResult:
        """
        Perceive input data and produce a structured representation.

        Args:
            input_data: The raw input (text, path, bytes, etc.)
            modality: Known modality, or auto-detect if None
            **context: Source, task, timestamp, etc.

        Returns:
            PerceptionResult with representation and confidence.
        """
        t0 = time.perf_counter()

        # Auto-detect modality if not specified
        if modality is None:
            modality = self._detect_modality(input_data)

        modality = Modality(modality) if isinstance(modality, str) else modality

        # Route to appropriate processor
        if modality == Modality.TEXT:
            result = self._text_processor.process(input_data, **context)
        elif modality == Modality.DOCUMENT:
            result = self._document_processor.process(input_data, **context)
        elif modality == Modality.STRUCTURED:
            result = self._structured_processor.process(input_data, **context)
        elif modality == Modality.IMAGE:
            if self._image_processor:
                result = self._image_processor.process(input_data, **context)
            else:
                result = self._image_fallback(input_data, **context)
        elif modality == Modality.AUDIO:
            if self._audio_processor:
                result = self._audio_processor.process(input_data, **context)
            else:
                result = self._audio_fallback(input_data, **context)
        elif modality == Modality.VIDEO:
            if self._video_processor:
                result = self._video_processor.process(input_data, **context)
            else:
                result = self._video_fallback(input_data, **context)
        else:
            result = self._generic_fallback(input_data, modality, **context)

        result.processing_latency_ms = (time.perf_counter() - t0) * 1000
        return result

    def _detect_modality(self, input_data: Any) -> Modality:
        """Attempt to detect the modality of input data."""
        if isinstance(input_data, str):
            # Could be text, path, URL, etc.
            if input_data.startswith(("http://", "https://")):
                return Modality.WEB_PAGE
            if re.search(r"\.(jpg|jpeg|png|gif|bmp|webp)$", input_data, re.I):
                return Modality.IMAGE
            if re.search(r"\.(mp3|wav|ogg|flac|aac)$", input_data, re.I):
                return Modality.AUDIO
            if re.search(r"\.(mp4|avi|mkv|mov|webm)$", input_data, re.I):
                return Modality.VIDEO
            if re.search(r"\.(pdf|doc|docx|xls|xlsx)$", input_data, re.I):
                return Modality.DOCUMENT
            # Check if it's structured data (JSON, CSV)
            stripped = input_data.strip()
            if stripped.startswith('{') or stripped.startswith('['):
                return Modality.STRUCTURED
            if ',' in stripped and chr(10) in stripped:
                return Modality.STRUCTURED
            # Default to text
            return Modality.TEXT
        elif isinstance(input_data, bytes):
            # Could be image, audio, etc. — need magic bytes
            return Modality.STRUCTURED
        elif isinstance(input_data, (dict, list)):
            return Modality.STRUCTURED
        return Modality.TEXT

    def _image_fallback(self, input_data: Any, **context) -> PerceptionResult:
        """Fallback for image perception when no model available."""
        rep = Representation(
            content=f"[Image: {input_data}]",
            representation_type=RepresentationType.PERCEPTUAL,
            quality=RepresentationQuality.LOW,
            confidence=0.3,
            source=context.get("source", "unknown"),
            modality="image",
            metadata={"note": "Image perception not available; metadata-only fallback"},
        )
        return PerceptionResult(
            representation=rep,
            confidence=PerceptionConfidence(overall=0.3,
                                           uncertainty_markers=["image_model_unavailable"]),
            detected_modalities=[Modality.IMAGE],
            extracted_features={"note": "No image model available"},
            missing_information=["visual_content", "objects", "text_in_image"],
        )

    def _audio_fallback(self, input_data: Any, **context) -> PerceptionResult:
        """Fallback for audio perception."""
        rep = Representation(
            content=f"[Audio: {input_data}]",
            representation_type=RepresentationType.PERCEPTUAL,
            quality=RepresentationQuality.LOW,
            confidence=0.3,
            source=context.get("source", "unknown"),
            modality="audio",
            metadata={"note": "Audio perception not available; metadata-only fallback"},
        )
        return PerceptionResult(
            representation=rep,
            confidence=PerceptionConfidence(overall=0.3,
                                           uncertainty_markers=["audio_model_unavailable"]),
            detected_modalities=[Modality.AUDIO],
            missing_information=["transcription", "speaker", "audio_features"],
        )

    def _video_fallback(self, input_data: Any, **context) -> PerceptionResult:
        """Fallback for video perception."""
        rep = Representation(
            content=f"[Video: {input_data}]",
            representation_type=RepresentationType.PERCEPTUAL,
            quality=RepresentationQuality.LOW,
            confidence=0.3,
            source=context.get("source", "unknown"),
            modality="video",
            metadata={"note": "Video perception not available; metadata-only fallback"},
        )
        return PerceptionResult(
            representation=rep,
            confidence=PerceptionConfidence(overall=0.3,
                                           uncertainty_markers=["video_model_unavailable"]),
            detected_modalities=[Modality.VIDEO],
            missing_information=["visual_content", "transcription", "temporal_events"],
        )

    def _generic_fallback(self, input_data: Any, modality: Modality,
                          **context) -> PerceptionResult:
        """Generic fallback for unknown modalities."""
        rep = Representation(
            content=str(input_data)[:500],
            representation_type=RepresentationType.PERCEPTUAL,
            quality=RepresentationQuality.UNKNOWN,
            confidence=0.2,
            source=context.get("source", "unknown"),
            modality=modality.value,
        )
        return PerceptionResult(
            representation=rep,
            confidence=PerceptionConfidence(overall=0.2),
            detected_modalities=[modality],
            missing_information=["modality_specific_features"],
        )


class TextPerceptionProcessor:
    """
    Text perception processor.

    Converts raw text into structured TextRepresentation with:
    - Entity extraction (simple NER via patterns)
    - Predicate extraction
    - Temporal marker detection
    - Certainty/hedging marker detection
    - Sentiment analysis (basic)
    - Question/declarative detection
    """

    # Simple entity patterns (phased approach — will be replaced by learned NER)
    ENTITY_PATTERNS = [
        (r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", "PERSON_ORGANIZATION"),
        (r"\b\d{4}-\d{2}-\d{2}\b", "DATE"),
        (r"\b\d{1,2}:\d{2}\b", "TIME"),
        (r"\b\d+\s+(?:hours?|days?|weeks?|months?|years?)\b", "DURATION"),
        (r"\b(?:the\s+)?(.+?)\s+(?:is|are|was|were|has|have)\s", "ENTITY_CLAIM"),
    ]

    CERTAIN_MARKERS = [
        "definitely", "certainly", "absolutely", "always", "never",
        "must", "will", "is", "are", "was", "were", "confirmed",
        "known", "established", "proven", "demonstrated",
    ]

    HEDGING_MARKERS = [
        "might", "could", "may", "possibly", "perhaps", "maybe",
        "probably", "likely", "apparently", "reportedly", "allegedly",
        "supposedly", "around", "approximately", "roughly", "about",
        "unclear", "uncertain", "unknown", "not sure", "seems",
        "appears", "suggests", "indicates",
    ]

    QUESTION_WORDS = {"what", "who", "why", "how", "which", "when", "where", "whose"}

    def process(self, text: str, **context) -> PerceptionResult:
        """
        Process text and produce a TextRepresentation.

        Args:
            text: The input text
            **context: Source, task, timestamp, etc.

        Returns:
            PerceptionResult with structured text representation.
        """
        t0 = time.perf_counter()

        text = text.strip()
        if not text:
            rep = TextRepresentation(
                content="",
                quality=RepresentationQuality.NONE,
                confidence=0.0,
                source=context.get("source", "unknown"),
            )
            return PerceptionResult(
                representation=rep,
                confidence=PerceptionConfidence(overall=0.0),
                detected_modalities=[Modality.TEXT],
                missing_information=["content"],
            )

        # Extract linguistic features
        entities = self._extract_entities(text)
        predicates = self._extract_predicates(text)
        temporal_markers = self._extract_temporal_markers(text)
        certainty_markers = self._extract_certainty_markers(text)
        sentiment = self._basic_sentiment(text)
        is_question = self._is_question(text)
        is_declarative = self._is_declarative(text)

        # Determine quality based on content richness
        quality = self._assess_quality(
            text, entities, predicates, temporal_markers, certainty_markers
        )

        # Build representation
        rep = TextRepresentation(
            content=text[:2000],  # Truncate very long inputs
            representation_type=RepresentationType.PERCEPTUAL,
            quality=quality,
            confidence=self._confidence_from_quality(quality),
            source=context.get("source", "unknown"),
            modality="text",
            entities=entities,
            predicates=predicates,
            temporal_markers=temporal_markers,
            certainty_markers=certainty_markers,
            sentiment=sentiment,
            language=self._detect_language(text),
            is_question=is_question,
            is_declarative=is_declarative,
        )

        # Extract additional features
        features = {
            "word_count": len(text.split()),
            "character_count": len(text),
            "sentence_count": text.count(".") + text.count("?") + text.count("!"),
            "entity_count": len(entities),
            "has_temporal": len(temporal_markers) > 0,
            "has_certainty": len(certainty_markers) > 0,
        }

        # Identify missing information
        missing = []
        if not entities:
            missing.append("no_entities_detected")
        if not predicates:
            missing.append("no_predicates_detected")
        if not temporal_markers and self._has_time_indicators(text):
            missing.append("temporal_markers_not_extracted")

        confidence = PerceptionConfidence(
            overall=self._confidence_from_quality(quality),
            modality_confidence={"text": self._confidence_from_quality(quality)},
            quality_markers=self._quality_markers(quality),
            uncertainty_markers=self._uncertainty_markers(text, certainty_markers),
            provenance=context.get("source", "text_perception"),
        )

        latency = (time.perf_counter() - t0) * 1000

        return PerceptionResult(
            representation=rep,
            confidence=confidence,
            detected_modalities=[Modality.TEXT],
            extracted_features=features,
            missing_information=missing,
            processing_latency_ms=latency,
        )

    def _extract_entities(self, text: str) -> list[dict[str, Any]]:
        """Extract entities from text using patterns (temporary scaffolding)."""
        entities = []
        for pattern, entity_type in self.ENTITY_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append({
                    "text": match.group(0),
                    "type": entity_type,
                    "start": match.start(),
                    "end": match.end(),
                    "method": "pattern",
                })
        # Deduplicate
        seen = set()
        unique = []
        for e in entities:
            key = (e["text"].lower(), e["type"])
            if key not in seen:
                seen.add(key)
                unique.append(e)
        return unique

    def _extract_predicates(self, text: str) -> list[str]:
        """Extract predicates (actions/states) from text."""
        predicates = []
        # Find copula constructions
        for match in re.finditer(
            r"\b(is|are|was|were|has|have|had|will|would|could|might|may)\s+(.+?)(?:\.|,|$|;)", 
            text, re.IGNORECASE
        ):
            predicate = match.group(2).strip()
            if len(predicate) >= 2 and len(predicate.split()) <= 10:
                predicates.append(predicate)
        return predicates[:10]  # Limit

    def _extract_temporal_markers(self, text: str) -> list[str]:
        """Extract temporal markers."""
        markers = []
        temporal_patterns = [
            r"\b\d{4}\b",                    # Years
            r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", # Dates
            r"\b\d{4}-\d{2}-\d{2}\b",       # ISO dates
            r"\b(?:on|in|at|during|before|after)\s+.+(?:\d|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
            r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
            r"\b(?:today|yesterday|tomorrow|now|then|currently|previously|recently|soon|later)",
            r"\b\d+\s+(?:hours?|days?|weeks?|months?|years?)\s+(?:ago|old|later|earlier)",
        ]
        for pattern in temporal_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                markers.append(match.group(0))
        return list(set(markers))[:10]

    def _extract_certainty_markers(self, text: str) -> list[str]:
        """Extract certainty/hedging markers."""
        text_lower = text.lower()
        markers = []
        for marker in self.CERTAIN_MARKERS:
            if marker in text_lower:
                markers.append(marker)
        for marker in self.HEDGING_MARKERS:
            if marker in text_lower:
                markers.append(marker)
        return markers

    def _basic_sentiment(self, text: str) -> Optional[float]:
        """Very basic sentiment estimation (temporary)."""
        text_lower = text.lower()
        positive_words = {"good", "great", "excellent", "amazing", "wonderful", 
                         "best", "better", "positive", "success", "happy", "love"}
        negative_words = {"bad", "terrible", "awful", "worst", "worse", 
                         "negative", "failure", "sad", "hate", "poor", "wrong"}
        pos = sum(1 for w in positive_words if w in text_lower)
        neg = sum(1 for w in negative_words if w in text_lower)
        if pos + neg == 0:
            return None
        return (pos - neg) / (pos + neg)

    def _is_question(self, text: str) -> bool:
        """Check if text is a question."""
        text = text.strip()
        if not text:
            return False
        if text.endswith("?"):
            return True
        if re.match(r"\b(what|who|why|how|which|when|where)\b", text, re.I):
            return True
        return False

    def _is_declarative(self, text: str) -> bool:
        """Check if text is a declarative statement."""
        text = text.strip()
        if not text or text.endswith("?") or text.endswith("!"):
            return False
        if re.match(r"\b(what|who|why|how|which|when|where|is\s+there|are\s+there)\b", text, re.I):
            return False
        return len(text) > 10

    def _assess_quality(self, text: str, entities: list, predicates: list,
                       temporal: list, certainty: list) -> RepresentationQuality:
        """Assess the quality of the perception based on content richness."""
        score = 0
        if len(text.split()) > 5:
            score += 1
        if entities:
            score += min(len(entities), 3)
        if predicates:
            score += min(len(predicates), 2)
        if temporal:
            score += 1
        if certainty:
            score += 1

        if score >= 5:
            return RepresentationQuality.HIGH
        elif score >= 3:
            return RepresentationQuality.MODERATE
        elif score >= 1:
            return RepresentationQuality.LOW
        return RepresentationQuality.UNKNOWN

    def _confidence_from_quality(self, quality: RepresentationQuality) -> float:
        """Map quality to confidence."""
        mapping = {
            RepresentationQuality.HIGH: 0.85,
            RepresentationQuality.MODERATE: 0.65,
            RepresentationQuality.LOW: 0.45,
            RepresentationQuality.UNKNOWN: 0.35,
            RepresentationQuality.NONE: 0.0,
        }
        return mapping.get(quality, 0.5)

    def _quality_markers(self, quality: RepresentationQuality) -> list[str]:
        """Return quality marker descriptions."""
        markers = {
            RepresentationQuality.HIGH: ["rich_content", "entities_detected", "structured"],
            RepresentationQuality.MODERATE: ["adequate_content", "some_entities"],
            RepresentationQuality.LOW: ["limited_content"],
            RepresentationQuality.UNKNOWN: ["insufficient_content"],
            RepresentationQuality.NONE: ["empty"],
        }
        return markers.get(quality, [])

    def _uncertainty_markers(self, text: str, certainty_markers: list) -> list[str]:
        """Return uncertainty markers based on text analysis."""
        markers = []
        text_lower = text.lower()
        hedging = [m for m in self.HEDGING_MARKERS if m in text_lower]
        if hedging:
            markers.extend(hedging)
        if not text.strip():
            markers.append("empty_input")
        if len(text.split()) < 5:
            markers.append("very_short_text")
        return markers

    def _detect_language(self, text: str) -> str:
        """Very basic language detection (temporary — ASCII vs non-ASCII)."""
        if re.search(r"[^\x00-\x7F]", text):
            return "non_ascii"
        return "english"

    def _has_time_indicators(self, text: str) -> bool:
        """Check if text likely contains time information."""
        time_words = {"time", "date", "when", "day", "year", "hour", "minute", "second",
                     "morning", "afternoon", "evening", "night", "ago", "later", "soon"}
        text_lower = text.lower()
        return any(w in text_lower for w in time_words)


# Export for easy access
__all__ = [
    "PerceptionEngine",
    "PerceptionResult",
    "PerceptionConfidence",
    "Modality",
    "TextPerceptionProcessor",
]
