"""
Representation Layer — Phase 1: Representation Reset.

This is the core abstraction that converts raw input into structured
representations that the rest of the cognitive system operates on.

Key principle: The neural mesh should operate on REPRESENTATIONS, not
raw text or rules. Representations carry:
- Content (what was perceived/understood)
- Context (surrounding information)
- Confidence (how reliable this representation is)
- Provenance (where it came from)
- Uncertainty (what we don't know)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import time


class RepresentationQuality(str, Enum):
    """How reliable a representation is believed to be."""
    HIGH = "high"           # Strong evidence, high confidence
    MODERATE = "moderate"   # Decent evidence, some uncertainty
    LOW = "low"             # Weak evidence, significant uncertainty
    UNKNOWN = "unknown"     # Cannot assess quality
    NONE = "none"           # No representation available


class RepresentationType(str, Enum):
    """What kind of representation this is."""
    PERCEPTUAL = "perceptual"        # Direct sensory input
    SEMANTIC = "semantic"            # Meaning extracted from input
    CONCEPTUAL = "conceptual"        # Abstract concept
    RELATIONAL = "relational"        # Relationship between entities
    CONTEXTUAL = "contextual"        # Context/situation
    PREDICTIVE = "predictive"        # Prediction about future
    HYPOTHETICAL = "hypothetical"    # 가설/추론
    EVIDENTIAL = "evidential"        # Evidence bearing on a claim
    PROCEDURAL = "procedural"        # How to do something


@dataclass
class EmbeddingVector:
    """
    A learned embedding vector with metadata.

    This is the primitive unit of representation. Everything else
    builds on top of embeddings.
    """
    vector: list[float]
    dimension: int = 0
    model_id: str = "unknown"
    quality: RepresentationQuality = RepresentationQuality.UNKNOWN
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.dimension:
            self.dimension = len(self.vector)

    def cosine_similarity(self, other: "EmbeddingVector") -> float:
        """Compute cosine similarity with another vector."""
        if not self.vector or not other.vector:
            return 0.0
        dot = sum(a * b for a, b in zip(self.vector, other.vector))
        norm_a = sum(a * a for a in self.vector) ** 0.5
        norm_b = sum(b * b for b in other.vector) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def euclidean_distance(self, other: "EmbeddingVector") -> float:
        """Compute Euclidean distance to another vector."""
        if not self.vector or not other.vector:
            return float('inf')
        return sum((a - b) ** 2 for a, b in zip(self.vector, other.vector)) ** 0.5


@dataclass
class Representation:
    """
    A structured representation of perceived/understood content.

    This is the base class for all representations in the cognitive system.
    Every module should produce and consume Representations, not raw data.
    """
    content: str
    representation_type: RepresentationType = RepresentationType.SEMANTIC
    quality: RepresentationQuality = RepresentationQuality.UNKNOWN
    confidence: float = 0.5
    source: str = ""
    modality: str = "text"
    created_at: float = field(default_factory=time.time)
    embedding: Optional[EmbeddingVector] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: f"rep_{int(time.time() * 1000000)}_{id(object())}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "type": self.representation_type.value,
            "quality": self.quality.value,
            "confidence": self.confidence,
            "source": self.source,
            "modality": self.modality,
            "created_at": self.created_at,
            "has_embedding": self.embedding is not None,
            "metadata": self.metadata,
        }

    def with_confidence(self, confidence: float) -> "Representation":
        """Return a copy with updated confidence."""
        result = self.__class__(
            content=self.content,
            representation_type=self.representation_type,
            quality=self.quality,
            confidence=confidence,
            source=self.source,
            modality=self.modality,
            created_at=self.created_at,
            embedding=self.embedding,
            metadata=self.metadata.copy(),
            id=self.id,
        )
        return result

    def with_embedding(self, embedding: EmbeddingVector) -> "Representation":
        """Return a copy with an embedding attached."""
        result = self.__class__(
            content=self.content,
            representation_type=self.representation_type,
            quality=self.quality,
            confidence=self.confidence,
            source=self.source,
            modality=self.modality,
            created_at=self.created_at,
            embedding=embedding,
            metadata=self.metadata.copy(),
            id=self.id,
        )
        return result


@dataclass
class TextRepresentation(Representation):
    """
    A representation of text with linguistic structure.

    Extracted linguistic features:
    - Entities mentioned
    - Key predicates/actions
    - Temporal markers
    - Sentiment/valence
    - Certainty markers
    """
    entities: list[dict[str, Any]] = field(default_factory=list)
    predicates: list[str] = field(default_factory=list)
    temporal_markers: list[str] = field(default_factory=list)
    certainty_markers: list[str] = field(default_factory=list)
    sentiment: Optional[float] = None  # -1 to +1
    language: str = "unknown"
    is_question: bool = False
    is_declarative: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({
            "entities": self.entities,
            "predicates": self.predicates,
            "temporal_markers": self.temporal_markers,
            "certainty_markers": self.certainty_markers,
            "sentiment": self.sentiment,
            "language": self.language,
            "is_question": self.is_question,
            "is_declarative": self.is_declarative,
        })
        return d


@dataclass
class MultimodalRepresentation(Representation):
    """
    A representation combining multiple modalities.

    Supports fusion of text, image, audio, video, etc.
    Each modality contribution is tracked separately for provenance.
    """
    modality_sources: list[dict[str, Any]] = field(default_factory=list)
    fused: bool = False
    alignment_quality: float = 0.5

    def add_modality_source(self, modality: str, content: str,
                            confidence: float, metadata: dict | None = None):
        """Add a modality contribution to this representation."""
        self.modality_sources.append({
            "modality": modality,
            "content": content,
            "confidence": confidence,
            "metadata": metadata or {},
            "timestamp": time.time(),
        })

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({
            "modality_sources": self.modality_sources,
            "fused": self.fused,
            "alignment_quality": self.alignment_quality,
        })
        return d


class RepresentationPipeline:
    """
    A pipeline that converts raw input into representations.

    This is the primary entry point for perception → representation.
    Subclasses implement specific pipelines for different modalities.
    """

    def __init__(self, name: str = "base"):
        self.name = name

    def process(self, raw_input: Any, **context) -> Representation:
        """
        Convert raw input into a representation.

        Args:
            raw_input: The raw input (text, image bytes, audio, etc.)
            **context: Additional context (source, task, etc.)

        Returns:
            A Representation of the input.
        """
        raise NotImplementedError

    def batch_process(self, inputs: list[Any], **context) -> list[Representation]:
        """Process multiple inputs."""
        return [self.process(item, **context) for item in inputs]


class RepresentationGateway:
    """
    Gateway that validates and enriches representations before they
    enter the cognitive system.

    This replaces the old bridge.py concept with a more general
    representation-focused interface.
    """

    def __init__(self, min_confidence: float = 0.3):
        self.min_confidence = min_confidence
        self.rejected: list[dict] = []

    def accept(self, representation: Representation) -> bool:
        """
        Check if a representation meets minimum quality standards.

        Returns True if the representation should enter the cognitive system.
        """
        if representation.confidence < self.min_confidence:
            self.rejected.append({
                "representation_id": representation.id,
                "reason": f"confidence {representation.confidence:.3f} below threshold {self.min_confidence}",
                "timestamp": time.time(),
            })
            return False
        return True

    def enrich(self, representation: Representation,
               context: dict | None = None) -> Representation:
        """
        Enrich a representation with additional context and metadata.

        This is where we add:
        - Task context
        - Source reliability
        - Temporal context
        - Related representations
        """
        enriched_meta = representation.metadata.copy()
        if context:
            enriched_meta["input_context"] = context
        enriched_meta["enriched_at"] = time.time()

        return representation.__class__(
            content=representation.content,
            representation_type=representation.representation_type,
            quality=representation.quality,
            confidence=representation.confidence,
            source=representation.source,
            modality=representation.modality,
            created_at=representation.created_at,
            embedding=representation.embedding,
            metadata=enriched_meta,
            id=representation.id,
        )

    def reject_count(self) -> int:
        return len(self.rejected)

    def clear_rejected(self):
        self.rejected.clear()
