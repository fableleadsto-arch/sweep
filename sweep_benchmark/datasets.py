"""Deterministic dataset generation for the Sweep benchmark (Phase 3).

Six groups, each with >=100 cases.  Every case carries:
  - group / family / difficulty
  - query + evidence
  - expected (canonical answer)
  - mode: one of
      int      -> expected is an exact integer (parsed from router output)
      yesno    -> expected in {"yes","no"} (decision mapping)
      claim    -> expected in {"supported","refuted","unknown"}
      pair     -> expected in {"consistent","contradiction","unknown"}
      set      -> expected is a sorted tuple of ints

Ground truth is computed *independently in Python* at generation time.
Synthetic entity names (zorps, fleeps, ...) are used so no knowledge base
can memorise the answers.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# Synthetic vocabulary (never used by Sweep's KB / training data)
# ──────────────────────────────────────────────────────────────────────
THINGS = ["zorp", "fleep", "gloop", "blix", "quark", "troon", "wibble", "daxel"]
VERBS = ["active", "running", "locked", "open", "charged", "enabled", "connected", "awake"]
COLORS = ["blue", "green", "red", "yellow"]
NEG_PREFIX = ["not ", "never ", "isn't "]

REL_WORDS = ["taller", "faster", "heavier", "older"]


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


# ──────────────────────────────────────────────────────────────────────
# Group 1: Basic Logic
# ──────────────────────────────────────────────────────────────────────
def gen_arithmetic(rng: random.Random, n: int) -> list[dict]:
    out = []
    ops = [("plus", "+", lambda a, b: a + b),
           ("times", "*", lambda a, b: a * b),
           ("minus", "-", lambda a, b: a - b),
           ("divided by", "/", lambda a, b: a // b)]
    for i in range(n):
        op_name, _, fn = rng.choice(ops)
        if op_name == "divided by":
            b = rng.randint(2, 12)
            a = b * rng.randint(2, 20)
        else:
            a = rng.randint(5, 250)
            b = rng.randint(3, 90)
        exp = fn(a, b)
        q = f"What is {a} {op_name} {b}?"
        out.append(_case("basic_logic", "arithmetic", "easy", q, [], int(exp), "int"))
    return out


def gen_sequences(rng: random.Random, n: int) -> list[dict]:
    out = []
    for i in range(n):
        start = rng.randint(0, 30)
        step = rng.randint(-12, 12)
        if step == 0:
            step = 1
        seq = [start + step * k for k in range(4)]
        nxt = seq[-1] + step
        q = f"What comes next: {', '.join(map(str, seq))}, ?"
        out.append(_case("basic_logic", "sequence", "easy", q, [], int(nxt), "int"))
    return out


def gen_boolean(rng: random.Random, n: int) -> list[dict]:
    out = []
    ops = {"and": lambda a, b: a and b,
           "or": lambda a, b: a or b,
           "xor": lambda a, b: a != b,
           "nand": lambda a, b: not (a and b),
           "nor": lambda a, b: not (a or b)}
    for i in range(n):
        a = rng.random() < 0.5
        b = rng.random() < 0.5
        op = rng.choice(list(ops))
        res = ops[op](a, b)
        q = f"{str(a).lower()} {op} {str(b).lower()}"
        # engine canonical boolean output is yes/no (router 'true'->yes, 'false'->no)
        out.append(_case("basic_logic", "boolean", "easy", q, [],
                         "yes" if res else "no", "yesno"))
    for i in range(n // 3):
        a = rng.random() < 0.5
        res = not a
        q = f"not {str(a).lower()}"
        out.append(_case("basic_logic", "boolean", "easy", q, [],
                         "yes" if res else "no", "yesno"))
    return out


def _fmt_set(vals) -> str:
    return "{" + ", ".join(map(str, sorted(vals))) + "}"


def gen_sets(rng: random.Random, n: int) -> list[dict]:
    out = []
    for i in range(n):
        kind = rng.random()
        a = rng.sample(range(1, 12), rng.randint(2, 4))
        b = rng.sample(range(1, 12), rng.randint(2, 4))
        if kind < 0.4:
            res = sorted(set(a) | set(b))
            q = f"What is the union of {_fmt_set(a)} and {_fmt_set(b)}?"
            out.append(_case("basic_logic", "set_union", "easy", q, [], tuple(res), "set"))
        elif kind < 0.7:
            res = sorted(set(a) & set(b))
            q = f"What is the intersection of {_fmt_set(a)} and {_fmt_set(b)}?"
            out.append(_case("basic_logic", "set_intersection", "easy", q, [], tuple(res), "set"))
        else:
            # subset question: choose sub as subset or not
            inter = sorted(set(a) & set(b))
            if rng.random() < 0.5:
                sub = inter if inter else [min(b)]
                ans = "yes"
            else:
                # force not-a-subset: add an element not in b
                extra = rng.randint(50, 80)
                while extra in set(b):
                    extra = rng.randint(50, 80)
                sub = sorted(set(a) | {extra})
                ans = "no"
            q = f"Is {_fmt_set(sub)} a subset of {_fmt_set(b)}?"
            out.append(_case("basic_logic", "set_subset", "medium", q, [], ans, "yesno",
                             expected_raw=ans))
    return out


def gen_number_theory(rng: random.Random, n: int) -> list[dict]:
    out = []

    def is_prime(x: int) -> bool:
        if x < 2:
            return False
        for d in range(2, int(x ** 0.5) + 1):
            if x % d == 0:
                return False
        return True

    for i in range(n):
        x = rng.randint(2, 200)
        ans = is_prime(x)
        q = f"Is {x} prime?"
        out.append(_case("basic_logic", "prime", "easy", q, [],
                         "yes" if ans else "no", "yesno", expected_raw=str(ans).lower()))
    return out


# ──────────────────────────────────────────────────────────────────────
# Group 2: Multi-Step Reasoning
# ──────────────────────────────────────────────────────────────────────
def gen_syllogisms(rng: random.Random, n: int) -> list[dict]:
    """Inline chained syllogisms: All A are B. All B are C. Are all A C?"""
    out = []
    words = ["zorbs", "greeps", "bloops", "quarks", "wibbles", "daxels", "troons", "fleeps"]
    while len(out) < n:
        a, b, c = rng.sample(words, 3)
        # vary question polarity: "Are all A C?" sometimes invalid conclusion
        if rng.random() < 0.7:
            # valid: All A are B. All B are C.  => A -> C
            q = f"All {a} are {b}. All {b} are {c}. Are all {a} {c}?"
            out.append(_case("multi_step", "chain_syllogism", "medium", q, [], "yes", "yesno",
                             expected_raw="yes"))
        else:
            # invalid: conclusion reversed (All C are A) does not follow
            q = f"All {a} are {b}. All {b} are {c}. Are all {c} {a}?"
            out.append(_case("multi_step", "chain_syllogism_fallacy", "hard", q, [], "unknown", "yesno",
                             expected_raw="unknown"))
    return out


def gen_transitivity(rng: random.Random, n: int) -> list[dict]:
    out = []
    names = [("Alice", "Bob", "Carol"), ("Alex", "Ben", "Chris"),
             ("Amy", "Brian", "Cathy"), ("Aaron", "Beth", "Cole")]
    while len(out) < n:
        x, y, z = rng.choice(names)
        rel = rng.choice(REL_WORDS)
        if rng.random() < 0.75:
            q = f"{x} is {rel} than {y}. {y} is {rel} than {z}. Is {x} {rel} than {z}?"
            out.append(_case("multi_step", "transitivity", "medium", q, [], "yes", "yesno",
                             expected_raw="yes"))
        else:
            # missing link between chains — cannot conclude
            w = rng.choice(["Dana", "Eve", "Frank", "Gina"])
            q = f"{x} is {rel} than {y}. {w} is {rel} than {z}. Is {x} {rel} than {z}?"
            out.append(_case("multi_step", "transitivity_gap", "hard", q, [], "unknown", "yesno",
                             expected_raw="unknown"))
    return out


def gen_no_syllogisms(rng: random.Random, n: int) -> list[dict]:
    """All A are B. No B are C. Are any A C?  (no)"""
    out = []
    words = ["fimps", "quuxes", "wibbles", "glorps", "blerts", "snorps", "klaxes", "morphs"]
    while len(out) < n:
        a, b, c = rng.sample(words, 3)
        q = f"All {a} are {b}. No {b} are {c}. Are any {a} {c}?"
        out.append(_case("multi_step", "neg_syllogism", "hard", q, [], "no", "yesno",
                         expected_raw="no"))
    return out


# ──────────────────────────────────────────────────────────────────────
# Claim-verification helpers (used by ambiguity / evidence / adversarial)
# ──────────────────────────────────────────────────────────────────────
def _claim_sup(rng: random.Random) -> dict:
    t = rng.choice(THINGS)
    v = rng.choice(VERBS)
    claim = f"The {t} is {v}."
    evidence = f"The {t} is {v} and running smoothly."
    return _case("evidence", "claim_support", "easy", claim, [evidence],
                 "supported", "claim", query_is_claim=True)


def _claim_ref(rng: random.Random) -> dict:
    t = rng.choice(THINGS)
    v = rng.choice(VERBS)
    claim = f"The {t} is {v}."
    evidence = f"The {t} is not {v}."
    return _case("evidence", "claim_refute", "easy", claim, [evidence],
                 "refuted", "claim", query_is_claim=True)


def _claim_unknown(rng: random.Random) -> dict:
    t = rng.choice(THINGS)
    v = rng.choice(VERBS)
    claim = f"The {t} is {v}."
    other = rng.choice([x for x in COLORS])
    evidence = f"The {t} is {other} and quiet."
    return _case("ambiguity", "claim_irrelevant", "medium", claim, [evidence],
                 "unknown", "claim", query_is_claim=True)


# ──────────────────────────────────────────────────────────────────────
# Group 3: Ambiguity
# ──────────────────────────────────────────────────────────────────────
def gen_ambiguity(rng: random.Random, n: int) -> list[dict]:
    out = []
    # 1. irrelevant evidence -> unknown (correct abstention)
    for _ in range(n // 4):
        out.append(_claim_unknown(rng))
    # 2. modal hedging in evidence -> unknown
    for _ in range(n // 4):
        t = rng.choice(THINGS)
        v = rng.choice(VERBS)
        claim = f"The {t} is {v}."
        hedge = rng.choice(["might be", "could be", "may be", "possibly is"])
        evidence = f"The {t} {hedge} {v}."
        out.append(_case("ambiguity", "hedged_claim", "hard", claim, [evidence],
                         "unknown", "claim", query_is_claim=True))
    # 3. conflicting sources (2 evidence) -> unknown which is right
    for _ in range(n // 4):
        t = rng.choice(THINGS)
        v = rng.choice(VERBS)
        claim = f"The {t} is {v}."
        evidence = [f"Source A says the {t} is {v}.",
                    f"Source B says the {t} is not {v}."]
        out.append(_case("ambiguity", "conflicting_sources", "hard", claim, evidence,
                         "unknown", "claim", query_is_claim=True))
    # 4. ambiguous date ("around/between") with single date-ish evidence
    for _ in range(n // 4):
        t = rng.choice(THINGS)
        claim = f"The {t} was launched in 1990."
        year = rng.choice([1989, 1991, 1992])
        evidence = f"Records say the {t} was launched around {year}."
        out.append(_case("ambiguity", "uncertain_date", "hard", claim, [evidence],
                         "unknown", "claim", query_is_claim=True))
    return out


# ──────────────────────────────────────────────────────────────────────
# Group 4: Evidence Reasoning
# ──────────────────────────────────────────────────────────────────────
def gen_evidence(rng: random.Random, n: int) -> list[dict]:
    out = []
    for _ in range(n // 3):
        out.append(_claim_sup(rng))
    for _ in range(n // 3):
        out.append(_claim_ref(rng))
    # contradiction pairs / consistency pairs
    for _ in range(n // 3):
        t = rng.choice(THINGS)
        v = rng.choice(VERBS)
        if rng.random() < 0.6:
            evidence = [f"The {t} is {v}.", f"The {t} is not {v}."]
            expected = "contradiction"
        else:
            evidence = [f"The {t} is {v}.", f"The {t} is {v} and operational."]
            expected = "consistent"
        q = "Compare these two statements."
        out.append(_case("evidence", "statement_pair", "medium", q, evidence,
                         expected, "pair"))
    return out


# ──────────────────────────────────────────────────────────────────────
# Group 5: Adversarial
# ──────────────────────────────────────────────────────────────────────
def gen_adversarial(rng: random.Random, n: int) -> list[dict]:
    out = []
    # double negation traps (single evidence claim)
    for _ in range(n // 5):
        t = rng.choice(THINGS)
        v = rng.choice(VERBS)
        claim = f"The {t} is {v}."
        evidence = f"The {t} is not not {v}."
        out.append(_case("adversarial", "double_negation", "hard", claim, [evidence],
                         "supported", "claim", query_is_claim=True))
    # negated evidence against positive claim
    for _ in range(n // 5):
        t = rng.choice(THINGS)
        v = rng.choice(VERBS)
        claim = f"The {t} is {v}."
        evidence = f"It is false that the {t} is {v}."
        out.append(_case("adversarial", "explicit_negation", "medium", claim, [evidence],
                         "refuted", "claim", query_is_claim=True))
    # distractors: unrelated sentences padded into evidence, but signal present
    for _ in range(n // 5):
        t = rng.choice(THINGS)
        v = rng.choice(VERBS)
        claim = f"The {t} is {v}."
        d1, d2 = rng.sample([x for x in COLORS], 2)
        evidence = [f"The {t} is {d1}.", f"The {t} is {v}.", f"The {t} is {d2}."]
        out.append(_case("adversarial", "distractor_padding", "hard", claim, evidence,
                         "supported", "claim", query_is_claim=True))
    # numbers differ -> contradiction
    for _ in range(n // 5):
        t = rng.choice(THINGS)
        a = rng.randint(10, 90)
        b = a + rng.randint(5, 30)
        evidence = [f"Report 1 says the {t} count was {a}.", f"Report 2 says the {t} count was {b}."]
        q = "Compare these two statements."
        out.append(_case("adversarial", "numeric_conflict", "medium", q, evidence,
                         "contradiction", "pair"))
    # false premises — statement claims about impossible/made-up conditions
    for _ in range(n // 5):
        t = rng.choice(THINGS)
        v = rng.choice(VERBS)
        claim = f"Because the {t} is {v}, the {t} is {v}."
        evidence = [f"The {t} is {v}."]
        out.append(_case("adversarial", "circular_claim", "hard", claim, evidence,
                         "supported", "claim", query_is_claim=True))
    return out


# ──────────────────────────────────────────────────────────────────────
# Group 6: Generalization (novel structures / novel content)
# ──────────────────────────────────────────────────────────────────────
def gen_generalization(rng: random.Random, n: int) -> list[dict]:
    out = []
    # 1. long arithmetic with 3 operands
    for _ in range(n // 7):
        a, b, c = [rng.randint(20, 900) for _ in range(3)]
        op = rng.choice(["+", "-", "*"])
        if op == "+":
            res = a + b + c
            q = f"What is {a} + {b} + {c}?"
        elif op == "-":
            res = a - b - c
            q = f"What is {a} - {b} - {c}?"
        else:
            res = a * b * c
            q = f"What is {a} * {b} * {c}?"
        out.append(_case("generalization", "multi_term_arithmetic", "hard", q, [], int(res), "int"))
    # 2. sequences with bigger steps / negatives (novel instantiations)
    for _ in range(n // 7):
        start = rng.randint(-200, -20)
        step = rng.randint(-50, -1) if rng.random() < 0.5 else rng.randint(15, 60)
        seq = [start + step * k for k in range(4)]
        nxt = seq[-1] + step
        q = f"What comes next: {', '.join(map(str, seq))}, ?"
        out.append(_case("generalization", "novel_sequence", "hard", q, [], int(nxt), "int"))
    # 3. longer 3-premise chain syllogisms
    words = ["vexes", "yotz", "krell", "squibs", "pilts", "gronks", "zwaps", "muntz", "hobbs", "loams"]
    for _ in range(n // 7):
        a, b, c, d = rng.sample(words, 4)
        q = f"All {a} are {b}. All {b} are {c}. All {c} are {d}. Are all {a} {d}?"
        out.append(_case("generalization", "deep_syllogism", "hard", q, [], "yes", "yesno",
                         expected_raw="yes"))
    # 4. 3-hop transitivity (4 people)
    for _ in range(n // 7):
        rel = rng.choice(REL_WORDS)
        names = [("Amy", "Ben", "Cole", "Dan"), ("Ada", "Bo", "Cy", "Dee"),
                 ("Ana", "Bob", "Cid", "Don")]
        a, b, c, d = rng.choice(names)
        q = f"{a} is {rel} than {b}. {b} is {rel} than {c}. {c} is {rel} than {d}. Is {a} {rel} than {d}?"
        out.append(_case("generalization", "deep_transitivity", "hard", q, [], "yes", "yesno",
                         expected_raw="yes"))
    # 5. abstract set operations with novel large sets
    for _ in range(n // 7):
        a = rng.sample(range(1, 100), 5)
        b = rng.sample(range(1, 100), 5)
        if rng.random() < 0.5:
            res = sorted(set(a) | set(b))
            q = f"What is the union of {_fmt_set(a)} and {_fmt_set(b)}?"
        else:
            res = sorted(set(a) & set(b))
            q = f"What is the intersection of {_fmt_set(a)} and {_fmt_set(b)}?"
        out.append(_case("generalization", "novel_sets", "hard", q, [], tuple(res), "set"))
    # 6. novel-predicate claim verification (content never in any KB)
    for _ in range(n // 7):
        t = rng.choice(["xylophone", "quantum relay", "tachyon drive", "neural lattice",
                        "plasma vane", "cryo module"])
        v = rng.choice(["calibrated", "synchronized", "quenched", "ionized", "polarized"])
        if rng.random() < 0.5:
            claim = f"The {t} is {v}."
            evidence = f"The {t} is {v} per diagnostic log 7."
            exp = "supported"
        else:
            claim = f"The {t} is {v}."
            evidence = f"The {t} is not {v} per diagnostic log 7."
            exp = "refuted"
        out.append(_case("generalization", "novel_claim", "medium", claim, [evidence],
                         exp, "claim", query_is_claim=True))
    # 7. abstract boolean expressions with brackets (novel syntax)
    for _ in range(n // 7):
        a = rng.random() < 0.5
        b = rng.random() < 0.5
        res = not (a or b)
        q = f"not ({str(a).lower()} or {str(b).lower()})"
        out.append(_case("generalization", "nested_boolean", "hard", q, [],
                         "yes" if res else "no", "yesno"))
    return out


# ──────────────────────────────────────────────────────────────────────
# Case builder + determinism helpers
# ──────────────────────────────────────────────────────────────────────
_counter = {"n": 0}


def _case(group, family, difficulty, query, evidence, expected, mode,
          expected_raw=None, query_is_claim=False) -> dict:
    _counter["n"] += 1
    return {
        "id": f"{group}_{_counter['n']:04d}",
        "group": group,
        "family": family,
        "difficulty": difficulty,
        "query": query,
        "evidence": list(evidence),
        "expected": expected if isinstance(expected, tuple) else str(expected),
        "mode": mode,
        "query_is_claim": query_is_claim,
    }


GENERATORS = [
    ("basic_logic", gen_arithmetic, 40),
    ("basic_logic", gen_sequences, 25),
    ("basic_logic", gen_boolean, 24),
    ("basic_logic", gen_sets, 20),
    ("basic_logic", gen_number_theory, 15),
    ("multi_step", gen_syllogisms, 45),
    ("multi_step", gen_transitivity, 45),
    ("multi_step", gen_no_syllogisms, 15),
    ("ambiguity", gen_ambiguity, 110),
    ("evidence", gen_evidence, 110),
    ("adversarial", gen_adversarial, 110),
    ("generalization", gen_generalization, 110),
]


def generate(seed: int = 20260903, per_group_min: int = 100) -> list[dict]:
    rng = _rng(seed)
    _counter["n"] = 0
    all_cases: list[dict] = []
    counts: dict[str, int] = {}
    for group, genfn, n in GENERATORS:
        cases = genfn(rng, n)
        counts[group] = counts.get(group, 0) + len(cases)
        all_cases.extend(cases)
    # guarantee per-group minimums
    missing = {g: max(0, per_group_min - c) for g, c in counts.items()}
    extra: dict[str, int] = {}
    if any(missing.values()):
        for group, n in missing.items():
            if n <= 0:
                continue
            fn = {"basic_logic": gen_arithmetic,
                  "multi_step": gen_syllogisms,
                  "ambiguity": gen_ambiguity,
                  "evidence": gen_evidence,
                  "adversarial": gen_adversarial,
                  "generalization": gen_generalization}[group]
            cases = fn(rng, n + 5)[:n]
            all_cases.extend(cases)
            counts[group] += len(cases)
    # shuffle deterministically, keep ids stable
    idx = list(range(len(all_cases)))
    rng.shuffle(idx)
    ordered = [all_cases[i] for i in idx]
    for i, c in enumerate(ordered):
        c["id"] = f"sb_{i:04d}"
    summary = {g: sum(1 for c in ordered if c["group"] == g) for g in sorted(counts)}
    return ordered, summary


def save(path: str, seed: int = 20260903) -> dict:
    cases, summary = generate(seed=seed)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "1.0.0",
        "seed": seed,
        "generator": "sweep_benchmark.datasets",
        "per_group": summary,
        "total": len(cases),
        "cases": cases,
    }
    Path(path).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return payload


def load(path: str) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["cases"]


if __name__ == "__main__":
    payload = save("benchmark/datasets/suite_v1.json")
    print(json.dumps(payload["per_group"]))
    print("total:", payload["total"])
