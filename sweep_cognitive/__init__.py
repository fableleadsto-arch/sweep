"""
SWEEP Cognitive Architecture — Phase 1+.

An original cognitive architecture combining:
- Human-inspired computational principles (hierarchical representations,
  associative memory, working memory, prediction, abstraction)
- Astra-style agent engineering (adaptive compute, tool orchestration,
  persistent state, verification, long-horizon execution)
- Existing SWEEP strengths (evidence consensus, contradiction detection,
  uncertainty handling, benchmark infrastructure)

The neural mesh is primarily a REPRESENTATIONAL AND COGNITIVE SUBSTATE,
not a rule engine. Explicit logic sits above or alongside learned
representations where it provides real benefit.

Phases:
    Phase 0: Architecture audit (complete)
    Phase 1: Representation reset (in progress)
    Phase 2+: See SWEEP_PHASE0_AUDIT.md for roadmap
"""

__version__ = "1.0.0"
__phase__ = "1"

from .representation import (
    Representation,
    TextRepresentation,
    MultimodalRepresentation,
    RepresentationQuality,
    EmbeddingVector,
)

from .perception import (
    PerceptionEngine,
    PerceptionResult,
    Modality,
    PerceptionConfidence,
)

from .knowledge import (
    KnowledgeBase,
    KnowledgeEntry,
    KnowledgeType,
    KnowledgeConfidence,
    KnowledgeSource,
)

__all__ = [
    "Representation",
    "TextRepresentation",
    "MultimodalRepresentation",
    "RepresentationQuality",
    "EmbeddingVector",
    "PerceptionEngine",
    "PerceptionResult",
    "Modality",
    "PerceptionConfidence",
    "KnowledgeBase",
    "KnowledgeEntry",
    "KnowledgeType",
    "KnowledgeConfidence",
    "KnowledgeSource",
]
