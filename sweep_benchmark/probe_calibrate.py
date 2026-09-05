"""Calibration probe — determines which case families Sweep answers deterministically
when live network retrieval is disabled. Used during design; not part of the final run."""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sweep_neural_mesh.neurons.cortex import ReasoningCortex

c = ReasoningCortex(enable_ml=False)
# Offline mode: disable live network retrieval paths
c.retrieve_live_knowledge = lambda q: None
c._try_live_knowledge = lambda q, e, t0: None
c.web_research = lambda *a, **k: None

def q(q_, ev=None):
    t0 = time.perf_counter()
    r = c.reason(query=q_, evidence=ev or [])
    dt = time.perf_counter() - t0
    exp = {k: str(v)[:80] for k, v in (r.explanation_data or {}).items()}
    return f"{dt:6.2f}s dec={r.decision:10s} conf={r.confidence:.2f} | {str(r.reasoning)[:160]} | expl={exp}"

CASES = [
    # warmup (cheap router hits)
    ("warm_arith", "What is 1 plus 1?", []),
    ("warm_seq", "What comes next: 1, 2, 3, ?", []),
    ("warm_bool", "true and true", []),
    # arithmetic variants
    ("arith_add", "What is 23 plus 19?", []),
    ("arith_minus", "What is 50 minus 17?", []),
    ("arith_mul", "What is 23 times 4?", []),
    ("arith_div", "What is 36 divided by 4?", []),
    ("arith_multi", "What is 12 plus 5 times 2?", []),
    # sequence
    ("seq_arith", "What comes next: 3, 6, 9, 12, ?", []),
    ("seq_geo", "What comes next: 2, 4, 8, 16, ?", []),
    ("seq_odd", "What comes next: 1, 3, 5, 7, ?", []),
    # boolean
    ("bool_and", "true and false", []),
    ("bool_or", "true or false", []),
    ("bool_not", "not true", []),
    ("bool_nor", "false nor false", []),
    # sets
    ("set_union", "What is the union of {1, 2} and {2, 3}?", []),
    ("set_inter", "What is the intersection of {1, 2, 3} and {2, 3, 4}?", []),
    ("set_sub_yes", "Is {1, 2} a subset of {1, 2, 3}?", []),
    ("set_sub_no", "Is {1, 2, 4} a subset of {1, 2, 3}?", []),
    # syllogism inline, synthetic words
    ("syl_chain", "All zorbs are greeps. All greeps are bloops. Are all zorbs bloops?", []),
    ("syl_neg", "No fimps are quuxes. All fimps are wibbles. Are any wibbles quuxes?", []),
    ("syl_sing", "All daxels are meeps. A troon is a daxel. Is a troon a meep?", []),
    # transitivity
    ("trans_tall", "Alex is taller than Ben. Ben is taller than Chris. Is Alex taller than Chris?", []),
    ("trans_fast", "A is faster than B. B is faster than C. Is A faster than C?", []),
    # modus ponens inline synthetic
    ("mp_1", "If a fleep is active then the zorch spins. The fleep is active. Is the zorch spinning?", []),
    # classification with evidence
    ("cls_yes", "Is a troon a meep?", ["A troon is a meep."]),
    ("cls_no", "Is a zorp a quark?", ["A zorp is not a quark."]),
    ("cls_unc", "Is a zorp a quark?", ["A zorp is blue."]),
    # evidence verdicts single
    ("ev_sup", "Is the zorp active?", ["The zorp is active."]),
    ("ev_ref", "Is the zorp active?", ["The zorp is not active."]),
    ("ev_irr", "Is the zorp active?", ["The zorp is blue."]),
    # contradiction pair
    ("contra_1", "Do these two statements contradict each other?", ["The zorp is active.", "The zorp is not active."]),
    ("contra_cons", "Are these two statements consistent?", ["The zorp is active.", "The zorp is active and running."]),
    # evidence values equal/different
    ("ev_vals", "Evidence A says value=5. Evidence B says value=8. Do the evidence values match?", ["Evidence A says value=5.", "Evidence B says value=8."]),
    # double negation adversarial
    ("dbl_neg", "Is it true that the zorp is not inactive?", ["The zorp is inactive."]),
    ("adv_false", "Given that 2 plus 2 equals 5, is 2 plus 2 equal to 4?", []),
    # ambiguity abstention
    ("amb_unknown", "Did the troon pass the test?", ["The results were not reported."]),
    ("amb_contra", "Is the zorp active?", ["Source A says the zorp is active.", "Source B says the zorp is not active."]),
]

for name, q_, ev in CASES:
    try:
        print(f"{name:12s} {q(q_, ev)}")
    except Exception as e:
        print(f"{name:12s} ERROR: {e}")
