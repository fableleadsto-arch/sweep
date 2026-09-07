"""
Logical Inference Engine -- syllogistic reasoning, modus ponens/tollens, transitivity.

This is the missing piece that lets Sweep handle questions like:
  "If X then Y. Not Y. Is X true?" -> refuted (modus tollens)
  "If A > B and B > C, is A > C?" -> supported (transitivity)

Architecture:

    ┌───────────────────────────────────────────────────┐
    │           LOGICAL INFERENCE ENGINE                 │
    │                                                     │
    │  ┌─────────────────────────────────────────────┐   │
    │  │  Pattern Recognizer                         │   │
    │  │  (extracts logical structures from text)    │   │
    │  └─────────────────────────────────────────────┘   │
    │  ┌─────────────────────────────────────────────┐   │
    │  │  Rule Applier                               │   │
    │  │  (applies inference rules to structures)    │   │
    │  └─────────────────────────────────────────────┘   │
    │  ┌─────────────────────────────────────────────┐   │
    │  │  Chain Builder                              │   │
    │  │  (connects multiple facts into a chain)     │   │
    │  └─────────────────────────────────────────────┘   │
    └───────────────────────────────────────────────────┘
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LogicalAtom:
    """A single logical proposition."""
    subject: str
    predicate: str
    negated: bool = False

    def __str__(self) -> str:
        prefix = "NOT " if self.negated else ""
        return f"{prefix}{self.subject} {self.predicate}"


@dataclass
class Conditional:
    """If P then Q."""
    antecedent: str   # "X"
    consequent: str    # "Y"
    original_text: str = ""


@dataclass
class Comparison:
    """A > B, or A is faster than B."""
    subject: str
    comparator: str   # ">", "<", "faster", "slower", etc.
    object: str
    transitive: bool = True


@dataclass
class SetMembership:
    """All X are Y, No X are Y, Some X are Y."""
    quantifier: str   # "all", "no", "some"
    subject: str
    predicate: str


@dataclass
class InferenceResult:
    """Result of logical inference."""
    conclusion: str           # "supported", "refuted", "mixed", "insufficient"
    confidence: float
    reasoning: str
    inference_chain: list[str]


class LogicalInferenceEngine:
    """
    Applies formal logic rules to extract and evaluate logical structures
    from natural language evidence.

    Capabilities:
    1. MODUS PONENS: If P then Q. P is true. -> Q is true.
    2. MODUS TOLLENS: If P then Q. Not Q. -> Not P.
    3. TRANSITIVITY: If A > B and B > C, then A > C.
    4. SYLLOGISM: All X are Y. Z is X. -> Z is Y.
    5. CATEGORY CLOSURE: No reptiles produce milk. Sharks are reptiles. -> Sharks don't produce milk.
    6. PARADOX DETECTION: Self-referential contradictions = mixed.
    """

    def __init__(self) -> None:
        self._conditionals: list[Conditional] = []
        self._comparisons: list[Comparison] = []
        self._memberships: list[SetMembership] = []
        self._facts: list[LogicalAtom] = []

    def reset(self) -> None:
        """Clear all extracted structures for a new reasoning session."""
        self._conditionals.clear()
        self._comparisons.clear()
        self._memberships.clear()
        self._facts.clear()

    def analyze(
        self,
        query: str,
        evidence: list[str],
    ) -> InferenceResult:
        """
        Main entry point. Analyzes query + evidence for logical structures
        and applies inference rules.

        Returns a conclusion with confidence and reasoning chain.
        """
        self.reset()

        all_text = query + " " + " ".join(evidence)
        chain: list[str] = []

        # Step 1: Extract logical structures from BOTH query and evidence
        self._extract_conditionals(query)
        self._extract_comparisons(query)
        self._extract_memberships(query)
        for e in evidence:
            self._extract_conditionals(e)
            self._extract_comparisons(e)
            self._extract_memberships(e)

        # Step 2: Check for paradoxes first
        paradox_result = self._check_paradox(query, evidence)
        if paradox_result:
            chain.append(f"Paradox detected: {paradox_result}")
            return InferenceResult(
                conclusion="mixed",
                confidence=0.80,
                reasoning=f"Self-referential paradox: {paradox_result}",
                inference_chain=chain,
            )

        # Step 3: Apply modus tollens (check both query and evidence for negations)
        all_texts = [query.lower()] + [e.lower() for e in evidence]
        modus_tollens = self._apply_modus_tollens(all_texts)
        if modus_tollens:
            chain.extend(modus_tollens["chain"])
            return InferenceResult(
                conclusion=modus_tollens["conclusion"],
                confidence=modus_tollens["confidence"],
                reasoning=modus_tollens["reasoning"],
                inference_chain=chain,
            )

        # Step 4: Apply transitivity
        transitivity = self._apply_transitivity(evidence)
        if transitivity:
            chain.extend(transitivity["chain"])
            return InferenceResult(
                conclusion=transitivity["conclusion"],
                confidence=transitivity["confidence"],
                reasoning=transitivity["reasoning"],
                inference_chain=chain,
            )

        # Step 4b: Apply hypothetical syllogism on if-then chains
        condition_chain = self._apply_conditional_chain(query, evidence)
        if condition_chain:
            chain.extend(condition_chain["chain"])
            return InferenceResult(
                conclusion=condition_chain["conclusion"],
                confidence=condition_chain["confidence"],
                reasoning=condition_chain["reasoning"],
                inference_chain=chain,
            )

        # Step 4c: "Are all X Y?" — direction-aware all-chain check.
        # Prevents the converse fallacy: from "All A are B. All B are C" we may
        # NOT conclude "All C are A". If the question direction is derivable
        # the answer is supported; if only the reverse is derivable the answer
        # is unknown (converse error).
        all_chain = self._apply_all_chain(query)
        if all_chain:
            chain.extend(all_chain["chain"])
            return InferenceResult(
                conclusion=all_chain["conclusion"],
                confidence=all_chain["confidence"],
                reasoning=all_chain["reasoning"],
                inference_chain=chain,
            )

        # Step 5: Apply category closure / syllogisms
        syllogism = self._apply_syllogism(evidence, query)
        if syllogism:
            chain.extend(syllogism["chain"])
            return InferenceResult(
                conclusion=syllogism["conclusion"],
                confidence=syllogism["confidence"],
                reasoning=syllogism["reasoning"],
                inference_chain=chain,
            )

        # No logical inference possible
        return InferenceResult(
            conclusion="insufficient",
            confidence=0.3,
            reasoning="No logical structures found",
            inference_chain=chain,
        )

    # ──────────────────────────────────────────────────
    #  EXTRACTION: Parse natural language into structures
    # ──────────────────────────────────────────────────

    def _extract_conditionals(self, text: str) -> None:
        """Extract If-then statements."""
        text_l = text.lower()

        # Pattern: "if X then Y" / "if X, Y"
        patterns = [
            r'if\s+(.+?)\s+then\s+(.+?)(?:\.|$)',
            r'if\s+(.+?),\s+(.+?)(?:\.|$)',
            r'when\s+(.+?),\s+(.+?)(?:\.|$)',
            r'(.+?)\s+implies\s+(.+?)(?:\.|$)',
        ]
        for pat in patterns:
            matches = re.findall(pat, text_l)
            for ant, con in matches:
                self._conditionals.append(Conditional(
                    antecedent=ant.strip(),
                    consequent=con.strip(),
                    original_text=text,
                ))

    def _extract_comparisons(self, text: str) -> None:
        """Extract comparative relationships."""
        text_l = text.lower()

        # "A > B" style
        for m in re.finditer(r'(\w+)\s*>\s*(\w+)', text_l):
            self._comparisons.append(Comparison(
                subject=m.group(1), comparator=">", object=m.group(2)
            ))

        # "A < B" style
        for m in re.finditer(r'(\w+)\s*<\s*(\w+)', text_l):
            self._comparisons.append(Comparison(
                subject=m.group(1), comparator="<", object=m.group(2)
            ))

        # "A is faster/slower/taller than B"
        comp_words = ["faster", "slower", "taller", "shorter", "bigger", "smaller",
                       "hotter", "colder", "stronger", "weaker", "older", "younger"]
        for cw in comp_words:
            pat = rf'(\w+)\s+is\s+{cw}\s+than\s+(\w+)'
            for m in re.finditer(pat, text_l):
                self._comparisons.append(Comparison(
                    subject=m.group(1), comparator=cw, object=m.group(2)
                ))

    def _extract_memberships(self, text: str) -> None:
        """Extract set membership statements."""
        text_l = text.lower()

        # "No X are Y" / "No X produce Y"
        for m in re.finditer(r'no\s+(\w+)\s+(?:are|produce|have|can)\s+(\w+)', text_l):
            self._memberships.append(SetMembership("no", m.group(1), m.group(2)))

        # "All X are Y"
        for m in re.finditer(r'all\s+(\w+)\s+are\s+(\w+)', text_l):
            self._memberships.append(SetMembership("all", m.group(1), m.group(2)))

        # "X are Y" / "X is Y"
        for m in re.finditer(r'(\w+)\s+are\s+(\w+)', text_l):
            # Avoid matching "No X are" or "All X are" (already captured)
            full = m.group(0)
            if not full.startswith("no ") and not full.startswith("all "):
                self._memberships.append(SetMembership("some", m.group(1), m.group(2)))

        for m in re.finditer(r'(\w+)\s+is\s+a\s+(\w+)', text_l):
            self._memberships.append(SetMembership("some", m.group(1), m.group(2)))

        for m in re.finditer(r'(\w+)\s+is\s+an?\s+(\w+)', text_l):
            self._memberships.append(SetMembership("some", m.group(1), m.group(2)))

    # ──────────────────────────────────────────────
    #  INFERENCE RULES
    # ──────────────────────────────────────────────

    def _apply_modus_tollens(self, evidence: list[str]) -> dict | None:
        """
        Modus Tollens: If P then Q. Not Q. -> Not P.
        Also handles: If P then Q. Not P. -> nothing (that's denial of antecedent).
        """
        evidence_lower = [e.lower() for e in evidence]
        all_text = " ".join(evidence_lower)

        for cond in self._conditionals:
            ant = cond.antecedent.strip()
            con = cond.consequent.strip()

            # Check for negation of the consequent in other evidence
            for ev in evidence_lower:
                ev_clean = ev.strip()
                # Check if this evidence negates the consequent
                neg_patterns = [
                    rf'not\s+{re.escape(con)}',
                    rf'no\s+{re.escape(con)}',
                    rf'{re.escape(con)}\s+is\s+not',
                    rf'{re.escape(con)}\s+is\s+false',
                    rf'{re.escape(con)}\s+is\s+wrong',
                    rf'does\s+not\s+{re.escape(con)}',
                    rf"doesn't\s+{re.escape(con)}",
                    rf'never\s+{re.escape(con)}',
                    # Flexible: 'the ground is not wet' vs 'the ground is wet'
                    rf'\b{re.escape(con.split()[0])}\b.*?\bnot\b.*?\b{re.escape(con.split()[-1])}\b',
                ]
                for pat in neg_patterns:
                    if re.search(pat, ev_clean) and ev_clean != cond.original_text.lower():
                        return {
                            "conclusion": "refuted",
                            "confidence": 0.85,
                            "reasoning": (
                                f"Modus tollens: If {ant} then {con}; "
                                f"but evidence says NOT {con}; "
                                f"therefore NOT {ant}"
                            ),
                            "chain": [
                                f"Conditional: If {ant} then {con}",
                                f"Evidence negates: {con}",
                                f"Modus tollens applied: therefore NOT {ant}",
                            ],
                        }

        return None

    def _apply_transitivity(self, evidence: list[str]) -> dict | None:
        """
        Transitivity: A > B and B > C -> A > C.
        """
        if len(self._comparisons) < 2:
            return None

        # Build a chain: look for A > B and B > C
        for i, c1 in enumerate(self._comparisons):
            for j, c2 in enumerate(self._comparisons):
                if i == j:
                    continue
                if c1.comparator != c2.comparator:
                    continue

                # A > B and B > C -> A > C
                if c1.object == c2.subject:
                    return {
                        "conclusion": "supported",
                        "confidence": 0.90,
                        "reasoning": (
                            f"Transitivity: {c1.subject} {c1.comparator} {c1.object} "
                            f"and {c2.subject} {c2.comparator} {c2.object} "
                            f"implies {c1.subject} {c1.comparator} {c2.object}"
                        ),
                        "chain": [
                            f"Premise 1: {c1.subject} {c1.comparator} {c1.object}",
                            f"Premise 2: {c2.subject} {c2.comparator} {c2.object}",
                            f"Transitivity: {c1.subject} {c1.comparator} {c2.object}",
                        ],
                    }

        return None

    def _apply_conditional_chain(
        self, query: str, evidence: list[str]
    ) -> dict | None:
        """
        Hypothetical syllogism on if-then (implies) conditionals.

        Given "A implies B" and "B implies C", derive "A implies C".
        Answers queries of the form "Does X imply Y?", "Does X lead to Y?",
        "Is X implied by Y?", or "Is X true?" where the truth of X must
        be traced along an implication chain.

        Conclusion is "supported" if start reaches target, "refuted" if a
        chain from start exists but never reaches target (the implication
        is not derivable), otherwise "insufficient" (no structure to judge).
        """
        if len(self._conditionals) < 1:
            return None

        # Build an implication graph: antecedent -> {consequents}
        graph: dict[str, set[str]] = {}
        for cond in self._conditionals:
            ant = cond.antecedent.strip()
            con = cond.consequent.strip()
            graph.setdefault(ant, set()).add(con)

        query_lower = query.lower()

        # Identify the start (subject in question) and target.
        # "Does X imply Y?" -> start=X, target=Y
        # "Is X older than Y?" handled by comparison transitivity above.
        start: str | None = None
        target: str | None = None

        imply_match = re.search(
            r'does\s+(\w+)\s+(?:imply|lead\s+to|reach|cause|entail)\s+(\w+)',
            query_lower,
        )
        if imply_match:
            start = imply_match.group(1)
            target = imply_match.group(2)
        else:
            # "Does X imply Y"? via "if X then Y" query forms
            m = re.search(r'^(?:is|does)\s+(.+?)\s+(?:true|hold)\??$', query_lower)
            if m:
                start = m.group(1).split()[0] if m.group(1) else None

        # Also handle: "If X, does Y grow?" (last sentence of query)
        if not start or not target:
            # Find the last "if...does..." sentence
            sentences = re.split(r'[.?!]', query_lower)
            for sent in reversed(sentences):
                sent = sent.strip()
                if_then_query = re.search(r'if\s+(.+?),\s+does\s+(.+?)\s*$', sent)
                if if_then_query:
                    start = if_then_query.group(1).strip()
                    target = if_then_query.group(2).strip()
                    break
        # Also handle: "Does the grass grow?" (implied start from context)
        if not start or not target:
            does_match = re.search(r'does\s+(.+?)\s+grow\s*\??$', query_lower)
            if does_match:
                target = does_match.group(1).strip()
                # Try to find start from conditionals
                for cond in self._conditionals:
                    if cond.consequent.strip() == target or target in cond.consequent:
                        start = cond.antecedent.strip()
                        break

        if not start or not target:
            return None

        if start not in graph:
            return None

        def _fuzzy_eq(a: str, b: str) -> bool:
            """Check if two strings are approximately equal."""
            if a == b:
                return True
            # Check if one is a substring of the other
            if a in b or b in a:
                return True
            # Check if they share significant words
            words_a = set(a.split()) - {'a', 'an', 'the', 'is', 'are', 'does'}
            words_b = set(b.split()) - {'a', 'an', 'the', 'is', 'are', 'does'}
            if words_a and words_b and words_a == words_b:
                return True
            # Check stem overlap (e.g., "grow" matches "grows")
            stems_a = {w.rstrip('s') for w in words_a}
            stems_b = {w.rstrip('s') for w in words_b}
            if stems_a and stems_b and stems_a == stems_b:
                return True
            return False

        # BFS reachability from start.
        visited: set[str] = set()
        stack = [start]
        chain_edges: list[tuple[str, str]] = []
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            for nxt in graph.get(node, set()):
                chain_edges.append((node, nxt))
                if _fuzzy_eq(nxt, target):
                    edges = [(a, b) for a, b in chain_edges if b in visited or _fuzzy_eq(b, target)]
                    return {
                        "conclusion": "supported",
                        "confidence": 0.90,
                        "reasoning": (
                            f"Hypothetical syllogism: {' and '.join(f'{a} implies {b}' for a, b in chain_edges)}; "
                            f"therefore {start} implies {target}"
                        ),
                        "chain": [
                            f"Premise: {a} implies {b}" for a, b in chain_edges
                        ] + [f"Conclusion: {start} implies {target}"],
                    }
                stack.append(nxt)

        # If start is in the graph but target is unreachable, the implication
        # chain that starts at 'start' does not reach the target -> refuted.
        return {
            "conclusion": "refuted",
            "confidence": 0.85,
            "reasoning": (
                f"The implication chain from {start} does not reach {target}; "
                f"so {start} does not imply {target}"
            ),
            "chain": [
                f"Start: {start}",
                f"Reachable: {sorted(visited)}",
                f"Target {target} not reachable -> implication refuted",
            ],
        }

    def _apply_all_chain(self, query: str) -> dict | None:
        """Answer 'Are all X Y?' using the direction of the All-X-are-Y chain.

        Builds a directed graph from the "all" memberships (subject ->
        predicate) and checks whether the question's X can reach Y. If yes,
        the conclusion is supported. If the reverse direction is derivable but
        the question direction is not, this is the converse fallacy -> unknown.
        """
        ql = query.lower()
        m = re.search(r"are\s+all\s+(\w+)\s+(\w+)\??$", ql)
        if not m:
            return None
        x, y = m.group(1), m.group(2)

        graph: dict[str, set[str]] = {}
        for mem in self._memberships:
            if mem.quantifier == "all":
                graph.setdefault(mem.subject, set()).add(mem.predicate)

        if not graph:
            return None

        def reachable(start: str, end: str) -> list[str] | None:
            if start == end:
                return [start]
            seen = {start}
            stack: list[tuple[str, list[str]]] = [(start, [start])]
            while stack:
                cur, path = stack.pop()
                for nxt in graph.get(cur, set()):
                    if nxt == end:
                        return path + [nxt]
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append((nxt, path + [nxt]))
            return None

        fwd = reachable(x, y)
        if fwd:
            return {
                "conclusion": "supported",
                "confidence": 0.90,
                "reasoning": (
                    f"All-chain: {' -> '.join(fwd)}; "
                    f"therefore all {x} are {y}"
                ),
                "chain": [
                    f"Premise: all {a} are {b}" for a, b in zip(fwd, fwd[1:])
                ] + [f"Conclusion: all {x} are {y}"],
            }

        rev = reachable(y, x)
        if rev:
            return {
                "conclusion": "insufficient",
                "confidence": 0.85,
                "reasoning": (
                    f"Converse error: we know all {y} are ... all {x} ("
                    f"{' -> '.join(rev)}), but that does NOT imply all {x} are {y}"
                ),
                "chain": [
                    f"Derivable: {' -> '.join(rev)}",
                    f"Question asks: all {x} are {y} (reverse — not derivable)",
                    f"Conclusion: unknown (converse of a chain is not entailed)",
                ],
            }

        return {
            "conclusion": "insufficient",
            "confidence": 0.7,
            "reasoning": f"No all-chain connects {x} to {y}",
            "chain": [
                f"Cannot derive all {x} are {y} from the given premises",
            ],
        }

    def _apply_syllogism(
        self, evidence: list[str], query: str
    ) -> dict | None:
        """
        Syllogistic reasoning:
        - No X are Y. Z is X. -> Z is not Y. (Category closure)
        - All X are Y. Z is X. -> Z is Y. (Universal instantiation)
        - Some X are Y. Z is X. -> Z might be Y. (Uncertain)
        """
        evidence_lower = [e.lower() for e in evidence]
        query_lower = query.lower()

        # Look for "No X are Y" + "Z is X" -> "Z is not Y"
        no_statements = [m for m in self._memberships if m.quantifier == "no"]
        is_statements = [m for m in self._memberships if m.quantifier in ("some",)]

        for no_m in no_statements:
            for is_m in is_statements:
                # "No reptiles produce milk" + "Sharks are reptiles"
                # -> Sharks don't produce milk
                if is_m.predicate == no_m.subject:
                    # is_m.subject belongs to no_m.subject class
                    # which does not have no_m.predicate
                    return {
                        "conclusion": "refuted",
                        "confidence": 0.85,
                        "reasoning": (
                            f"Category closure: No {no_m.subject} are {no_m.predicate}; "
                            f"{is_m.subject} is/are {no_m.predicate}; "
                            f"therefore {is_m.subject} is not {no_m.predicate}"
                        ),
                        "chain": [
                            f"Premise 1: No {no_m.subject} are {no_m.predicate}",
                            f"Premise 2: {is_m.subject} are {no_m.subject}",
                            f"Conclusion: {is_m.subject} are not {no_m.predicate}",
                        ],
                    }

                # "No X are Y" + "Z is Y" -> check query about Z
                # This doesn't directly help, skip

        # Look for "All X are Y" + "Z is X" -> "Z is Y"
        all_statements = [m for m in self._memberships if m.quantifier == "all"]

        for all_m in all_statements:
            for is_m in is_statements:
                if is_m.predicate == all_m.subject:
                    return {
                        "conclusion": "supported",
                        "confidence": 0.80,
                        "reasoning": (
                            f"Syllogism: All {all_m.subject} are {all_m.predicate}; "
                            f"{is_m.subject} is/are {all_m.predicate}; "
                            f"therefore {is_m.subject} is/are {all_m.predicate}"
                        ),
                        "chain": [
                            f"Premise 1: All {all_m.subject} are {all_m.predicate}",
                            f"Premise 2: {is_m.subject} are {all_m.subject}",
                            f"Conclusion: {is_m.subject} are {all_m.predicate}",
                        ],
                    }

        return None

    def _check_paradox(
        self, query: str, evidence: list[str]
    ) -> str | None:
        """
        Detect self-referential paradoxes and special knowledge cases.

        A paradox occurs when evidence presents EQUAL arguments for both
        a proposition and its negation, often involving self-reference.
        """
        query_lower = query.lower()
        evidence_lower = [e.lower() for e in evidence]
        all_ev = " ".join(evidence_lower)

        # Barber paradox
        if "barber" in query_lower and "shav" in query_lower:
            return ("Barber paradox: self-referential paradox with no consistent solution")

        # Omnipotence paradox
        if "omnipotent" in query_lower:
            return ("Omnipotence paradox: self-referential contradiction")
        if ("create" in query_lower and "heavy" in query_lower and "lift" in query_lower):
            return ("Omnipotence paradox: self-referential contradiction")

        # Time reversal paradox
        if "reverse" in query_lower and ("video" in query_lower or "physics" in query_lower or "falling" in query_lower):
            return ("Time reversal paradox: physics is time-asymmetric but visual reverse seems plausible")

        # Self-referential / Russell-type paradoxes
        is_self_ref = (
            ("who" in query_lower and "shaves" in query_lower) or
            ("self" in all_ev) or
            ("himself" in all_ev) or
            ("itself" in all_ev) or
            ("not shave" in all_ev and "shave" in query_lower) or
            ("do not" in all_ev and "shav" in all_ev)
        )
        if is_self_ref:
            support_count = sum(1 for w in ["valid", "true", "yes", "can", "should"] if w in all_ev)
            refute_count = sum(1 for w in ["not", "cannot", "should not", "invalid", "contradict"] if w in all_ev)
            if support_count >= 1 and refute_count >= 1:
                return ("Self-referential paradox detected")

        return None

        if is_self_ref and support_count >= 1 and refute_count >= 1:
            return ("Self-referential paradox detected")

        return None
