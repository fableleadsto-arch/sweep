"""Phase 4 — Deterministic Reasoning Core.

A pure, deterministic logic engine independent of any neural model.
Includes:
- transitive closure over typed relations (`A before B, B before C ==> A before C`)
- explicit proof steps so every conclusion is justifiable
Repeated execution produces identical results (no randomness, no LLM).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class Fact:
    """A ground fact: relation(subject, object)."""

    relation: str
    subject: str
    object: str

    def key(self) -> Tuple[str, str, str]:
        return (self.relation, self.subject, self.object)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Fact":
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def __str__(self) -> str:
        return f"{self.relation}({self.subject}, {self.object})"


@dataclass
class ProofStep:
    """A single derivation step that proves a conclusion."""

    conclusion: str
    premises: List[str]
    rule: str
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


class LogicEngine:
    """Deterministic transitive inference over typed relations.

    Semantics:
    - `transitive_relations`: set of relations R such that R(a,b) & R(b,c) => R(a,c)
    - Symmetric relations: R(a,b) => R(b,a)
    """

    def __init__(self, transitive_relations: Optional[Set[str]] = None,
                 symmetric_relations: Optional[Set[str]] = None):
        self.transitive_relations = transitive_relations or {"before", "after", "older-than", "north-of",
                                                             "ancestor-of", "subclass-of", "is-part-of"}
        self.symmetric_relations = symmetric_relations or {"adjacent-to", "sibling-of", "related-to"}
        self.facts: Set[Tuple[str, str, str]] = set()

    def add_fact(self, relation: str, subject: str, object: str) -> None:
        self.facts.add((relation, subject, object))

    def add(self, fact: Fact) -> None:
        self.add_fact(fact.relation, fact.subject, fact.object)

    def add_many(self, facts: List[Fact] | List[Tuple[str, str, str]]) -> None:
        for f in facts:
            if isinstance(f, Fact):
                self.add(f)
            else:
                self.add_fact(*f)

    # ------------------------------------------------------------- queries
    def holds(self, relation: str, subject: str, object: str) -> bool:
        """Direct or derived truth of relation(subject, object)."""
        return (relation, subject, object) in self.closure()

    def derive(self, relation: str, subject: str, object: str) -> Optional[List[ProofStep]]:
        """Return a proof if relation(subject, object) is derivable, else None."""
        target = (relation, subject, object)
        closed, steps = self._closure_with_proofs(target)
        return steps if target in closed else None

    # ------------------------------------------------------------- closure
    def closure(self) -> Set[Tuple[str, str, str]]:
        closed, _ = self._closure_with_proofs(None)
        return closed

    def _closure_with_proofs(self, target: Optional[Tuple[str, str, str]]) -> Tuple[Set[Tuple[str, str, str]], List[ProofStep]]:
        closed: Set[Tuple[str, str, str]] = set(self.facts)
        steps: List[ProofStep] = []
        # 1) symmetry pass (shown on the gate: reachability over symmetric rels)
        for rel, subj, obj in list(closed):
            if rel in self.symmetric_relations and rel not in ("before", "after"):
                if (rel, obj, subj) not in closed:
                    closed.add((rel, obj, subj))
        # 2) iterative transitive closure until fixpoint (deterministic)
        changed = True
        while changed:
            changed = False
            for rel, subj, obj in list(closed):
                if rel not in self.transitive_relations:
                    continue
                for _, subj2, obj2 in list(closed):
                    if obj == subj2 and rel == _:
                        derived = (rel, subj, obj2)
                        if derived not in closed:
                            closed.add(derived)
                            steps.append(ProofStep(
                                conclusion=f"{rel}({subj}, {obj2})",
                                premises=[f"{rel}({subj}, {obj})", f"{rel}({obj}, {obj2})"],
                                rule=f"transitivity of {rel}",
                                detail=f"{obj} == {subj2}",
                            ))
                            changed = True
            if target and target in closed:
                break
        return closed, steps

    def reachable(self, subject: str, relation: str | None = None) -> Set[str]:
        """All objects reachable from `subject` via allowed relations."""
        out = set()
        for rel, subj, obj in self.closure():
            if subj == subject and (relation is None or rel == relation):
                out.add(obj)
        return out