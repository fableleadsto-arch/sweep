"""Calibration probe #2 — claim-style verification, number theory, negation, corroboration."""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sweep_neural_mesh.neurons.cortex import ReasoningCortex

c = ReasoningCortex(enable_ml=False)
c.retrieve_live_knowledge = lambda q: None
c._try_live_knowledge = lambda q, e, t0: None
c.web_research = lambda *a, **k: None

def q(q_, ev=None):
    t0 = time.perf_counter()
    r = c.reason(query=q_, evidence=ev or [])
    dt = time.perf_counter() - t0
    exp = {k: str(v)[:80] for k, v in (r.explanation_data or {}).items()}
    return f"{dt:6.2f}s dec={r.decision:10s} conf={r.confidence:.2f} | {str(r.reasoning)[:170]} | {exp}"

CASES = [
    # warmup
    ("w1", "What is 1 plus 1?", []),
    # prime / number theory
    ("prime_yes", "Is 17 prime?", []),
    ("prime_no", "Is 21 prime?", []),
    ("even_yes", "Is 14 even?", []),
    ("gcd_q", "What is the gcd of 12 and 18?", []),
    ("factors_q", "What are the factors of 12?", []),
    # statement claims (single evidence)
    ("claim_sup", "The zorp is active.", ["The zorp is active and running."]),
    ("claim_ref", "The zorp is active.", ["The zorp is not active."]),
    ("claim_irr", "The zorp is active.", ["The zorp is blue and quiet."]),
    ("claim_corr", "The zorp is active.", ["The zorp is active in sector 4."]),
    # corroboration - 2 evidence, query about consistency
    ("cons_yes", "Do the two reports agree?", ["Report 1: revenue grew to 120.", "Report 2: revenue reached 120."]),
    ("cons_no", "Do the two reports agree?", ["Report 1: revenue was 120.", "Report 2: revenue was 95."]),
    ("cons_src", "Which source is more reliable?", ["Source A is an official government report.", "Source B is an anonymous blog."]),
    # negation
    ("neg_claim", "The zorp is not active.", ["The zorp is active."]),
    ("dblneg2", "It is not true that the zorp is not active.", ["The zorp is active."]),
    ("no_syl_inline", "No fimps are quuxes. A blip is a fimp. Is a blip a quux?", []),
    ("all_none", "All fimps are quuxes. No quuxes are wibbles. Is a fimp a wibble?", []),
    # larger arithmetic / ordering - generalization probe
    ("big_add", "What is 237 plus 419?", []),
    ("big_mul", "What is 47 times 13?", []),
    ("seq7", "What comes next: 5, 12, 19, 26, ?", []),
    ("cmp2", "Is 19 greater than 12?", []),
]

for name, q_, ev in CASES:
    try:
        print(f"{name:12s} {q(q_, ev)}")
    except Exception as e:
        print(f"{name:12s} ERROR: {e}")
