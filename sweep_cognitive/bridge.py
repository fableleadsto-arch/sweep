"""
Neural-to-Representation Bridge — Phase 1.

This module validates neural model outputs and converts them into
proper SWEEP representations. It replaces the old cognition/bridge.py
with a representation-focused interface.

Key principle: Neural outputs must be validated before they become
trusted representations. Low-confidence or malformed output is rejected
or downgraded, never silently trusted.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .representation import (
    Representation,
    TextRepresentation,
    EmbeddingVector,
    RepresentationQuality,
    RepresentationType,
)


@dataclass
class NeuralOutput:
    """Raw output from a neural model."""
    raw: str
    structured: dict[str, Any]
    confidence: float
    model_id: str = "unknown"
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "structured": self.structured,
            "confidence": self.confidence,
            "model_id": self.model_id,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }


@dataclass
class BridgeValidationResult:
    """Outcome of validating a neural output."""
    accepted: bool
    reason: str
    representation: Optional[Representation] = None
    warnings: list[str] = field(default_factory=list)
    rejected_at: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "has_representation": self.representation is not None,
            "warnings": self.warnings,
            "rejected_at": self.rejected_at,
        }


class NeuralBridge:
    """
    Validates neural outputs and converts them to SWEEP representations.

    This is the gateway through which neural model outputs enter the
    cognitive system. Every neural output must pass through here.
    """

    # Minimum confidence thresholds by output type
    MIN_CONFIDENCE = {
        "claim": 0.6,
        "evidence": 0.5,
        "entity": 0.5,
        "concept": 0.5,
        "relation": 0.5,
        "prediction": 0.4,
        "hypothesis": 0.3,
    }

    # Required fields by output type
    REQUIRED_FIELDS = {
        "claim": {"statement"},
        "evidence": {"content"},
        "entity": {"name"},
        "concept": {"name"},
        "relation": {"subject", "predicate", "object"},
        "prediction": {"content"},
        "hypothesis": {"statement"},
    }

    def __init__(self, min_confidence_override: float | None = None):
        self.min_confidence_override = min_confidence_override
        self.validation_log: list[dict] = []

    def validate(self, output: NeuralOutput, 
                 output_type: str = "claim") -> BridgeValidationResult:
        """
        Validate a neural output and convert to representation if accepted.

        Steps:
        1. Check output type is known
        2. Check required fields present
        3. Check confidence threshold
        4. Transform into Representation
        5. Log validation
        """
        t0 = time.time()
        warnings = []

        # Step 1: Check output type
        if output_type not in self.REQUIRED_FIELDS:
            return BridgeValidationResult(
                accepted=False,
                reason=f"unknown output_type '{output_type}'",
                rejected_at=t0,
                warnings=warnings,
            )

        # Step 2: Check structured output is a dict
        if not isinstance(output.structured, dict):
            return BridgeValidationResult(
                accepted=False,
                reason="structured output is not a dict",
                rejected_at=t0,
                warnings=warnings,
            )

        # Step 3: Check required fields
        required = self.REQUIRED_FIELDS[output_type]
        missing = required - set(output.structured.keys())
        if missing:
            return BridgeValidationResult(
                accepted=False,
                reason=f"missing required field(s): {sorted(missing)}",
                rejected_at=t0,
                warnings=warnings,
            )

        # Step 4: Check raw text is non-empty
        if not isinstance(output.raw, str) or not output.raw.strip():
            return BridgeValidationResult(
                accepted=False,
                reason="empty raw text",
                rejected_at=t0,
                warnings=warnings,
            )

        # Step 5: Check confidence threshold
        min_conf = self.min_confidence_override or self.MIN_CONFIDENCE.get(output_type, 0.5)
        if output.confidence < min_conf:
            warnings.append(
                f"confidence {output.confidence:.3f} below threshold {min_conf}"
            )
            # Still accept but mark quality as low
            quality = RepresentationQuality.LOW
        else:
            quality = RepresentationQuality.MODERATE

        # Step 6: Transform into representation
        representation = self._transform(output, output_type, quality, warnings)

        # Step 7: Log validation
        self.validation_log.append({
            "model_id": output.model_id,
            "output_type": output_type,
            "confidence": output.confidence,
            "accepted": True,
            "quality": quality.value,
            "timestamp": t0,
        })

        return BridgeValidationResult(
            accepted=True,
            reason="accepted",
            representation=representation,
            warnings=warnings,
        )

    def _transform(self, output: NeuralOutput, output_type: str,
                   quality: RepresentationQuality, 
                   warnings: list[str]) -> Representation:
        """Transform validated neural output into a Representation."""
        s = output.structured
        base_kwargs = {
            "source": output.model_id,
            "modality": "neural",
            "quality": quality,
            "metadata": {
                "neural_model": output.model_id,
                "raw_preview": output.raw[:200],
                "latency_ms": output.latency_ms,
                **output.metadata,
            },
        }

        if output_type == "claim":
            return TextRepresentation(
                content=s["statement"],
                representation_type=RepresentationType.EVIDENTIAL,
                confidence=output.confidence,
                is_declarative=True,
                **base_kwargs,
            )

        elif output_type == "evidence":
            return TextRepresentation(
                content=s["content"],
                representation_type=RepresentationType.EVIDENTIAL,
                confidence=output.confidence,
                **base_kwargs,
            )

        elif output_type == "entity":
            return TextRepresentation(
                content=f"Entity: {s['name']}",
                representation_type=RepresentationType.CONCEPTUAL,
                confidence=output.confidence,
                entities=[{"text": s["name"], "type": s.get("kind", "unknown")}],
                **base_kwargs,
            )

        elif output_type == "concept":
            return TextRepresentation(
                content=f"Concept: {s['name']}",
                representation_type=RepresentationType.CONCEPTUAL,
                confidence=output.confidence,
                **base_kwargs,
            )

        elif output_type == "relation":
            content = f"{s['subject']} {s['predicate']} {s['object']}"
            return TextRepresentation(
                content=content,
                representation_type=RepresentationType.RELATIONAL,
                confidence=output.confidence,
                predicates=[s["predicate"]],
                **base_kwargs,
            )

        elif output_type == "prediction":
            return TextRepresentation(
                content=s["content"],
                representation_type=RepresentationType.PREDICTIVE,
                confidence=output.confidence,
                **base_kwargs,
            )

        elif output_type == "hypothesis":
            return TextRepresentation(
                content=s["statement"],
                representation_type=RepresentationType.HYPOTHETICAL,
                confidence=output.confidence,
                **base_kwargs,
            )

        else:
            # Fallback: generic representation
            return Representation(
                content=output.raw[:500],
                representation_type=RepresentationType.SEMANTIC,
                confidence=output.confidence,
                **base_kwargs,
            )

    def rejection_rate(self) -> float:
        """Return the fraction of outputs that were rejected."""
        if not self.validation_log:
            return 0.0
        accepted = sum(1 for log in self.validation_log if log.get("accepted", False))
        return 1.0 - (accepted / len(self.validation_log))

    def clear_log(self):
        """Clear the validation log."""
        self.validation_log.clear()


class RepresentationGateway:
    """
    Higher-level gateway that combines validation with enrichment.

    This is the preferred entry point for neural outputs.
    """

    def __init__(self, min_confidence: float = 0.3):
        self.bridge = NeuralBridge()
        self.min_confidence = min_confidence

    def process(self, output: NeuralOutput, 
                output_type: str = "claim",
                context: dict | None = None) -> BridgeValidationResult:
        """
        Process a neural output through validation and enrichment.

        Args:
            output: The neural model output
            output_type: Type of output (claim, evidence, entity, etc.)
            context: Additional context for enrichment

        Returns:
            Validation result with representation if accepted.
        """
        # Validate
        result = self.bridge.validate(output, output_type)

        if not result.accepted:
            return result

        # Enrich with context if available
        if result.representation and context:
            enriched_meta = result.representation.metadata.copy()
            enriched_meta["input_context"] = context
            enriched_meta["gateway_processed_at"] = time.time()

            result.representation = result.representation.__class__(
                content=result.representation.content,
                representation_type=result.representation.representation_type,
                quality=result.representation.quality,
                confidence=result.representation.confidence,
                source=result.representation.source,
                modality=result.representation.modality,
                created_at=result.representation.created_at,
                embedding=result.representation.embedding,
                metadata=enriched_meta,
                id=result.representation.id,
            )

        return result

    def batch_process(self, outputs: list[NeuralOutput],
                      output_type: str = "claim",
                      context: dict | None = None) -> list[BridgeValidationResult]:
        """Process multiple neural outputs."""
        return [
            self.process(output, output_type, context)
            for output in outputs
        ]

    def stats(self) -> dict[str, Any]:
        """Return gateway statistics."""
        return {
            "total_validated": len(self.bridge.validation_log),
            "rejection_rate": self.bridge.rejection_rate(),
            "by_quality": {},
        }


__all__ = [
    "NeuralBridge",
    "RepresentationGateway",
    "NeuralOutput",
    "BridgeValidationResult",
]
