"""
Logic Handler — formal and informal reasoning tasks.

Handles:
  - Deductive reasoning (syllogisms, modus ponens, modus tollens)
  - Inductive reasoning (pattern completion, generalization)
  - Analogical reasoning (cross-domain mapping)
  - Classification (categorization, taxonomy)
  - Boolean logic (AND, OR, NOT, XOR, implications)
  - Set theory (subset, union, intersection, complement)
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class LogicResult:
    """Structured result from a logic handler."""
    answer: str
    confidence: float
    method: str
    reasoning_chain: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class LogicHandler:
    """Handles formal and informal logic reasoning tasks."""

    def process(self, query: str, evidence: list[str] | None = None) -> LogicResult:
        t0 = time.perf_counter()
        q = query.strip()
        ev = evidence or []

        # Try each reasoning method in order
        result = self._try_syllogism(q, ev, t0)
        if result:
            return result

        result = self._try_deduction(q, ev, t0)
        if result:
            return result

        result = self._try_induction(q, ev, t0)
        if result:
            return result

        result = self._try_analogy(q, ev, t0)
        if result:
            return result

        result = self._try_boolean_logic(q, t0)
        if result:
            return result

        result = self._try_set_theory(q, t0)
        if result:
            return result

        result = self._try_classification(q, ev, t0)
        if result:
            return result

        return LogicResult(
            answer="", confidence=0.0, method="none",
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    # -- Syllogisms -------------------------------------------------------

    def _try_syllogism(self, q: str, ev: list[str], t0: float) -> LogicResult | None:
        """Handle categorical syllogisms (All A are B, X is A -> X is B)."""
        # Normalize: strip punctuation so periods don't break regex
        q_clean = re.sub(r'[.?!]', ' ', q.lower())
        
        # Detect INVALID syllogisms: "All A are B. Some B are C. Are all A C?"
        # This is the fallacy of the undistributed middle
        some_are = re.findall(r'some\s+(\w+)\s+are\s+(\w+)', q_clean)
        if some_are and 'are all' in q_clean:
            # Check if the question asks about all A being C, where only SOME B are C
            for some_subj, some_cat in some_are:
                # If the question asks "Are all X [some_cat]?" and the evidence is "Some [some_subj] are [some_cat]"
                if re.search(rf'are\s+all\s+\w+\s+{some_cat}', q_clean):
                    return LogicResult(
                        answer=f"no, only some {some_subj} are {some_cat}",
                        confidence=0.90,
                        method="invalid_syllogism",
                        reasoning_chain=[
                            f"Premise: Some {some_subj} are {some_cat} (not all)",
                            f"Conclusion: Cannot conclude all are {some_cat}",
                            "Fallacy: undistributed middle",
                        ],
                        latency_ms=(time.perf_counter() - t0) * 1000,
                    )

        # Extract premises: "All A are B" (B can be multi-word like "living things")
        # Match up to double-space (sentence boundary) or next keyword
        all_are = re.findall(r'all\s+(\w+)\s+are\s+([\w ]+?)(?:\s{2,}|\s+(?:all|is|the)\b)', q_clean)
        if not all_are:
            all_are = re.findall(r'all\s+(\w+)\s+are\s+(\w+)', q_clean)

        # Extract the question: "is X a Y?" where Y can be multi-word
        is_match = re.search(r'\bis\s+(?:a\s+)?(\w+)\s+(?:a\s+)?([\w ]+?)(?:\s{2,}|\s*$)', q_clean)
        question_subj = is_match.group(1).strip() if is_match else None
        question_cat = is_match.group(2).strip() if is_match else None

        if all_are:
            premises = {}
            for subject, category in all_are:
                premises[subject.strip()] = category.strip()

            # Find the matching premise for the question subject
            if question_subj:
                # Check singular/plural matching for the subject
                matched_subj = None
                for key in premises:
                    if question_subj == key:
                        matched_subj = key
                    elif question_subj + 's' == key:
                        matched_subj = key
                    elif question_subj.rstrip('s') + 's' == key:
                        matched_subj = key
                    elif question_subj == key.rstrip('s'):
                        matched_subj = key
                if matched_subj:
                    # Follow the premise chain to find the final category
                    result_cat = premises[matched_subj]
                    chain_steps = [
                        f"Premise: All {matched_subj} are {result_cat}",
                        f"Premise: {question_subj} is a {matched_subj.rstrip('s')}",
                    ]
                    # Follow chain: if result_cat is also a key, keep following
                    visited = {matched_subj}
                    while result_cat in premises and result_cat not in visited:
                        visited.add(result_cat)
                        next_cat = premises[result_cat]
                        chain_steps.append(f"Premise: All {result_cat} are {next_cat}")
                        result_cat = next_cat
                    # Check if the question category matches the final category
                    if question_cat and (question_cat in result_cat or result_cat in question_cat):
                        chain_steps.append(f"Conclusion: {question_subj} is a {result_cat}")
                        return LogicResult(
                            answer=f"yes, {question_subj} is a {result_cat}",
                            confidence=0.95,
                            method="syllogism",
                            reasoning_chain=chain_steps,
                            latency_ms=(time.perf_counter() - t0) * 1000,
                        )

        # Pattern: "No A are B. X is A. Is X B?"
        no_are = re.findall(r'no\s+(\w+)\s+are\s+(\w+)', q_clean)
        is_a2 = re.findall(r'(\w+)\s+is\s+(?:a\s+)?(\w+)', q_clean)
        if no_are and is_a2:
            for no_subj, no_cat in no_are:
                for subj, cat in is_a2:
                    if cat == no_subj:
                        chain = [
                            f"Premise: No {no_subj} are {no_cat}",
                            f"Premise: {subj} is {no_subj}",
                            f"Conclusion: {subj} is not {no_cat}",
                        ]
                        return LogicResult(
                            answer=f"no, {subj} is not {no_cat}",
                            confidence=0.95,
                            method="syllogism",
                            reasoning_chain=chain,
                            latency_ms=(time.perf_counter() - t0) * 1000,
                        )

        return None

    # -- Deduction --------------------------------------------------------

    def _try_deduction(self, q: str, ev: list[str], t0: float) -> LogicResult | None:
        """Handle modus ponens, modus tollens, hypothetical syllogisms."""
        q_lower = q.lower()

        # Modus ponens: "If P then Q. P is true. Is Q true?" or "If P, Q."
        if_then = re.findall(r'if\s+(.+?)\s+then\s+(.+?)(?:[.?]|$)', q_lower)
        if not if_then:
            # Also handle "If P, Q." format
            if_then = re.findall(r'if\s+(.+?),\s+(.+?)(?:[.?]|$)', q_lower)
        if if_then:
            antecedent, consequent = if_then[0]
            antecedent_clean = antecedent.strip()
            consequent_clean = consequent.strip()
            
            # Check if antecedent is affirmed ONLY in evidence (not the query itself)
            # Exclude evidence that is the conditional statement itself
            antecedent_confirmed = False
            consequent_confirmed = False
            conditional_pattern = re.compile(r'if\s+.+?\s+(?:then\s+)?\s*.+', re.IGNORECASE)
            
            def _fuzzy_match(needle: str, haystack: str) -> bool:
                """Check if needle words appear in haystack (allows morphological variation)."""
                if needle in haystack:
                    return True
                # Check if core content words appear
                needle_words = set(needle.split()) - {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'it', 'that', 'this'}
                haystack_words = set(haystack.split())
                if len(needle_words) == 0:
                    return False
                # At least 70% of content words must appear
                matched = needle_words & haystack_words
                # Also check for word stems (e.g., "sounds" matches "sounding")
                for nw in needle_words:
                    for hw in haystack_words:
                        if nw.startswith(hw[:4]) or hw.startswith(nw[:4]):
                            matched.add(nw)
                            break
                return len(matched) / len(needle_words) >= 0.7
            
            for e in ev:
                e_lower = e.lower()
                # Skip if this evidence IS the conditional statement
                if conditional_pattern.match(e_lower.strip()):
                    if _fuzzy_match(consequent_clean, e_lower):
                        consequent_confirmed = True
                    continue
                # This is a separate statement - check for antecedent confirmation
                if _fuzzy_match(antecedent_clean, e_lower):
                    antecedent_confirmed = True
                if _fuzzy_match(consequent_clean, e_lower):
                    consequent_confirmed = True
            
            # Also check if antecedent is explicitly stated as true
            for e in ev:
                e_lower = e.lower()
                if conditional_pattern.match(e_lower.strip()):
                    continue
                if _fuzzy_match(antecedent_clean, e_lower) and any(w in e_lower for w in ["is true", "is occurring", "happened", "occurred", "is the case", "is sounding", "is raining"]):
                    antecedent_confirmed = True
            
            if antecedent_confirmed:
                # Valid modus ponens: antecedent is affirmed -> conclude consequent
                chain = [
                    f"If {antecedent_clean} then {consequent_clean}",
                    f"{antecedent_clean} is true (from evidence)",
                    f"Conclusion: {consequent_clean}",
                ]
                return LogicResult(
                    answer=consequent_clean,
                    confidence=0.90,
                    method="modus_ponens",
                    reasoning_chain=chain,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )
            elif consequent_confirmed and not antecedent_confirmed:
                # Affirming the consequent: only Q is confirmed, not P
                # This is a logical fallacy -- answer is unknown
                chain = [
                    f"If {antecedent_clean} then {consequent_clean}",
                    f"{consequent_clean} is confirmed (but not {antecedent_clean})",
                    f"Cannot conclude: affirming the consequent is a fallacy",
                ]
                return LogicResult(
                    answer="unknown",
                    confidence=0.85,
                    method="affirming_consequent_fallacy",
                    reasoning_chain=chain,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )

        # Modus tollens: "If P then Q. Not Q. Therefore not P."
        if if_then:
            antecedent, consequent = if_then[0]
            for e in ev + [q_lower]:
                if ("not" in e or "no" in e) and consequent.strip() in e:
                    chain = [
                        f"If {antecedent} then {consequent}",
                        f"Not {consequent}",
                        f"Conclusion: Not {antecedent}",
                    ]
                    return LogicResult(
                        answer=f"not {antecedent.strip()}",
                        confidence=0.85,
                        method="modus_tollens",
                        reasoning_chain=chain,
                        latency_ms=(time.perf_counter() - t0) * 1000,
                    )

        # Transitivity: "A > B. B > C. Therefore A > C."
        trans = re.findall(r'(\w+)\s*(?:is\s+)?(?:faster|better|greater|more|stronger)\s+than\s+(\w+)', q_lower)
        if len(trans) >= 2:
            a, b = trans[0]
            c, d = trans[1]
            if b == c:
                for word in ["faster", "better", "greater", "more", "stronger"]:
                    if word in q_lower:
                        chain = [
                            f"{a} is {word} than {b}",
                            f"{b} is {word} than {c}",
                            f"Conclusion: {a} is {word} than {c}",
                        ]
                        return LogicResult(
                            answer=f"{a} is {word} than {c}",
                            confidence=0.90,
                            method="transitivity",
                            reasoning_chain=chain,
                            latency_ms=(time.perf_counter() - t0) * 1000,
                        )

        return None

    # -- Induction --------------------------------------------------------

    def _try_induction(self, q: str, ev: list[str], t0: float) -> LogicResult | None:
        """Handle pattern completion and generalization."""
        q_lower = q.lower()

        # "What comes next: 2, 4, 6, 8, ?" -> 10
        # (signed numbers so negative steps/values work: "18, 9, 0, -9" -> -18)
        nums = re.findall(r'(-?\d+)', q_lower)
        if len(nums) >= 3 and '?' in q_lower:
            nums_int = [int(n) for n in nums]
            # Check arithmetic progression
            diffs = [nums_int[i+1] - nums_int[i] for i in range(len(nums_int)-1)]
            if len(set(diffs)) == 1:
                next_val = nums_int[-1] + diffs[0]
                chain = [f"Sequence: {nums_int}", f"Common difference: {diffs[0]}", f"Next: {next_val}"]
                return LogicResult(
                    answer=str(next_val),
                    confidence=0.95,
                    method="arithmetic_progression",
                    reasoning_chain=chain,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )
            # Check geometric progression (non-zero, constant ratio)
            if all(nums_int[i] != 0 and nums_int[0] != 0 and
                   abs(nums_int[i+1] / nums_int[i] - nums_int[1] / nums_int[0]) < 1e-9
                   for i in range(len(nums_int)-1)):
                ratio = nums_int[1] / nums_int[0]
                if float(ratio).is_integer():
                    next_val = int(nums_int[-1] * ratio)
                    chain = [f"Sequence: {nums_int}", f"Ratio: {ratio}", f"Next: {next_val}"]
                    return LogicResult(
                        answer=str(next_val),
                        confidence=0.90,
                        method="geometric_progression",
                        reasoning_chain=chain,
                        latency_ms=(time.perf_counter() - t0) * 1000,
                    )

        # "What is the pattern: ..."
        pattern_match = re.search(r'pattern.*?(-?\d[\d,.\s-]+)', q_lower)
        if pattern_match and '?' in q_lower:
            nums_str = re.findall(r'(-?\d+)', pattern_match.group(1))
            if len(nums_str) >= 2:
                nums_int = [int(n) for n in nums_str]
                diffs = [nums_int[i+1] - nums_int[i] for i in range(len(nums_int)-1)]
                if len(set(diffs)) == 1:
                    next_val = nums_int[-1] + diffs[0]
                    return LogicResult(
                        answer=str(next_val),
                        confidence=0.90,
                        method="pattern_completion",
                        latency_ms=(time.perf_counter() - t0) * 1000,
                    )

        return None

    # -- Analogy ----------------------------------------------------------

    def _try_analogy(self, q: str, ev: list[str], t0: float) -> LogicResult | None:
        """Handle analogical reasoning: A is to B as C is to D."""
        # Pattern: "A is to B as C is to ?"
        analogy = re.search(
            r'(\w+(?:\s+\w+)?)\s+is\s+to\s+(\w+(?:\s+\w+)?)\s+as\s+(\w+(?:\s+\w+)?)\s+is\s+to\s+\?',
            q.lower(),
        )
        if analogy:
            a, b, c = analogy.group(1).strip(), analogy.group(2).strip(), analogy.group(3).strip()
            # Find the relationship between a and b
            relationship = self._find_relationship(a, b, ev)
            if relationship:
                d = self._apply_relationship(c, relationship, ev)
                if d:
                    chain = [
                        f"{a} is to {b} as {c} is to ?",
                        f"Relationship: {a} -> {b} ({relationship})",
                        f"Apply: {c} -> {d}",
                    ]
                    return LogicResult(
                        answer=d,
                        confidence=0.80,
                        method="analogy",
                        reasoning_chain=chain,
                        latency_ms=(time.perf_counter() - t0) * 1000,
                    )

        # Pattern: "X is similar to Y because..."
        similar = re.search(r'(\w+)\s+is\s+(?:similar|like)\s+to\s+(\w+)', q.lower())
        if similar and ev:
            a, b = similar.group(1), similar.group(2)
            for e in ev:
                if "because" in e.lower() or "since" in e.lower() or "due to" in e.lower():
                    return LogicResult(
                        answer=e[:200],
                        confidence=0.70,
                        method="analogy",
                        reasoning_chain=[f"Analogy: {a} ~ {b}", f"Reason: {e[:100]}"],
                        latency_ms=(time.perf_counter() - t0) * 1000,
                    )

        return None

    def _find_relationship(self, a: str, b: str, ev: list[str]) -> str | None:
        """Find the relationship between two terms."""
        for e in ev:
            e_lower = e.lower()
            if a in e_lower and b in e_lower:
                if "cause" in e_lower or "because" in e_lower:
                    return "causal"
                if "part" in e_lower or "member" in e_lower:
                    return "meronymy"
                if "type" in e_lower or "kind" in e_lower:
                    return "hypernymy"
                if "like" in e_lower or "similar" in e_lower:
                    return "similarity"

        if b in ("animal", "plant", "object", "concept", "language", "country"):
            return "hypernymy"
        return "association"

    _ANALOGY_TABLE: dict[str, dict[str, str]] = {
        "puppy": {"dog": "puppy", "cat": "kitten", "horse": "foal", "cow": "calf",
                   "sheep": "lamb", "duck": "duckling", "swan": "cygnet"},
        "kitten": {"dog": "puppy", "cat": "kitten"},
        "foal": {"horse": "foal", "cow": "calf"},
        "lamb": {"sheep": "lamb", "goat": "kid"},
        "calf": {"cow": "calf", "elephant": "calf", "whale": "calf"},
        "son": {"father": "son", "mother": "son", "parent": "child"},
        "daughter": {"father": "daughter", "mother": "daughter", "parent": "child"},
        "wheel": {"car": "wheel", "bicycle": "wheel"},
        "engine": {"car": "engine", "airplane": "engine"},
    }

    def _apply_relationship(self, c: str, relationship: str, ev: list[str]) -> str | None:
        """Apply a relationship to find the answer."""
        if relationship == "hypernymy":
            for e in ev:
                if c in e.lower():
                    cat_match = re.search(rf'{c}\s+is\s+(?:a|an)\s+(\w+)', e.lower())
                    if cat_match:
                        return cat_match.group(1)

        c_lower = c.lower()
        for parent_key, mappings in self._ANALOGY_TABLE.items():
            if c_lower in mappings:
                result = mappings[c_lower]
                if result != c_lower:
                    return result

        if relationship == "hypernymy":
            common_children = {
                "dog": "puppy", "cat": "kitten", "horse": "foal",
                "cow": "calf", "sheep": "lamb", "duck": "duckling",
                "swan": "cygnet", "goat": "kid", "fox": "kit",
                "bear": "cub", "lion": "cub", "tiger": "cub",
                "eagle": "eaglet", "owl": "owlet",
            }
            if c_lower in common_children:
                return common_children[c_lower]

        return None

    # -- Boolean Logic ----------------------------------------------------

    def _try_boolean_logic(self, q: str, t0: float) -> LogicResult | None:
        """Handle boolean expressions: AND, OR, NOT, XOR, NAND, NOR, parens.

        Supports nested / parenthesized expressions such as
        "not (false or true)" via a small recursive descent evaluator.
        """
        q_lower = q.lower().strip().rstrip('.').strip()
        if not re.search(r'\b(true|false)\b', q_lower):
            return None
        if not re.search(r'\b(and|or|xor|nand|nor)\b|\bnot\b|\(', q_lower):
            return None

        # Tokenize: true/false literals, operators, parens
        tokens = re.findall(r'true|false|and|or|xor|nand|nor|not|\(|\)|[^\s]+', q_lower)
        if not tokens:
            return None

        pos = 0

        def peek():
            return tokens[pos] if pos < len(tokens) else None

        def parse_primary():
            nonlocal pos
            tok = peek()
            if tok == "not":
                pos += 1
                val, expr = parse_primary()
                return not val, f"not ({expr})"
            if tok == "(":
                pos += 1
                val, expr = parse_or()
                if peek() == ")":
                    pos += 1
                return val, f"({expr})"
            if tok in ("true", "false"):
                pos += 1
                return tok == "true", tok
            pos += 1
            return False, ""

        def parse_and():
            nonlocal pos
            val, expr = parse_primary()
            ops = []
            while peek() in ("and", "nand"):
                op = tokens[pos]
                pos += 1
                rhs, rexpr = parse_primary()
                if op == "and":
                    val = val and rhs
                else:
                    val = not (val and rhs)
                ops.append(f"{expr} {op} {rexpr}")
            return val, " and ".join(ops) if ops else expr

        def parse_or():
            nonlocal pos
            val, expr = parse_and()
            ops = []
            while peek() in ("or", "xor", "nor"):
                op = tokens[pos]
                pos += 1
                rhs, rexpr = parse_and()
                if op == "or":
                    val = val or rhs
                elif op == "xor":
                    val = val != rhs
                else:
                    val = not (val or rhs)
                ops.append(f"{expr} {op} {rexpr}")
            return val, " or ".join(ops) if ops else expr

        try:
            result, _ = parse_or()
        except Exception:
            return None
        # Ensure the whole expression was consumed
        if pos != len(tokens):
            return None

        chain = [f"{q_lower} = {str(result).lower()}"]
        return LogicResult(
            answer=str(result).lower(),
            confidence=0.99,
            method="boolean",
            reasoning_chain=chain,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    # -- Set Theory -------------------------------------------------------

    def _try_set_theory(self, q: str, t0: float) -> LogicResult | None:
        """Handle basic set theory questions."""
        q_lower = q.lower()

        # Union: "{1,2} union {3,4}" or "What is the union of {1,2} and {3,4}?"
        union = re.search(r'(?:union\s+of\s+)?\{([^}]+)\}\s*union\s*\{([^}]+)\}', q_lower)
        if not union:
            union = re.search(r'union\s+of\s+\{([^}]+)\}\s+and\s+\{([^}]+)\}', q_lower)
        if union:
            set_a = set(x.strip() for x in union.group(1).split(","))
            set_b = set(x.strip() for x in union.group(2).split(","))
            result = set_a | set_b
            if "how many" in q_lower:
                return LogicResult(
                    answer=str(len(result)),
                    confidence=0.95,
                    method="set_union_count",
                    reasoning_chain=[
                        f"A = {{{', '.join(sorted(set_a))}}}",
                        f"B = {{{', '.join(sorted(set_b))}}}",
                        f"|A U B| = {len(result)}",
                    ],
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )
            sorted_result = sorted(result, key=lambda x: (x.isdigit(), int(x) if x.isdigit() else 0, x))
            formatted = "{" + ", ".join(sorted_result) + "}"
            return LogicResult(
                answer=formatted,
                confidence=0.95,
                method="set_union",
                reasoning_chain=[
                    f"A = {{{', '.join(sorted(set_a))}}}",
                    f"B = {{{', '.join(sorted(set_b))}}}",
                    f"A U B = {formatted}",
                ],
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        # Intersection
        intersect = re.search(r'intersection\s+of\s+\{([^}]+)\}\s+and\s+\{([^}]+)\}', q_lower)
        if intersect:
            set_a = set(x.strip() for x in intersect.group(1).split(","))
            set_b = set(x.strip() for x in intersect.group(2).split(","))
            result = set_a & set_b
            if "how many" in q_lower:
                return LogicResult(
                    answer=str(len(result)),
                    confidence=0.95,
                    method="set_intersection_count",
                    reasoning_chain=[f"|A n B| = {len(result)}"],
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )
            sorted_result = sorted(result, key=lambda x: (x.isdigit(), int(x) if x.isdigit() else 0, x))
            formatted = "{" + ", ".join(sorted_result) + "}"
            return LogicResult(
                answer=formatted,
                confidence=0.95,
                method="set_intersection",
                reasoning_chain=[f"A n B = {formatted}"],
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        # Subset
        subset = re.search(r'is\s+\{([^}]+)\}\s+a?\s*subset\s+of\s+\{([^}]+)\}', q_lower)
        if subset:
            set_a = set(x.strip() for x in subset.group(1).split(","))
            set_b = set(x.strip() for x in subset.group(2).split(","))
            is_subset = set_a <= set_b
            return LogicResult(
                answer="yes" if is_subset else "no",
                confidence=0.95,
                method="set_subset",
                reasoning_chain=[f"A subset B: {is_subset}"],
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        return None

    # -- Classification ---------------------------------------------------

    def _try_classification(self, q: str, ev: list[str], t0: float) -> LogicResult | None:
        """Handle categorization and classification tasks."""
        q_lower = q.lower()

        # "Is X a Y?" -- check against evidence
        is_a = re.match(r'is\s+(.+?)\s+(?:a|an)\s+(.+?)\??$', q_lower)
        if is_a and ev:
            entity = is_a.group(1).strip()
            category = is_a.group(2).strip()

            for e in ev:
                e_lower = e.lower()
                if entity in e_lower:
                    if category in e_lower or f"a {category}" in e_lower or f"an {category}" in e_lower:
                        return LogicResult(
                            answer=f"yes, {entity} is a {category}",
                            confidence=0.85,
                            method="classification",
                            reasoning_chain=[f"Evidence confirms: {entity} is a {category}"],
                            latency_ms=(time.perf_counter() - t0) * 1000,
                        )
                    if f"not a {category}" in e_lower or f"not an {category}" in e_lower:
                        return LogicResult(
                            answer=f"no, {entity} is not a {category}",
                            confidence=0.85,
                            method="classification",
                            reasoning_chain=[f"Evidence denies: {entity} is not a {category}"],
                            latency_ms=(time.perf_counter() - t0) * 1000,
                        )

        return None
