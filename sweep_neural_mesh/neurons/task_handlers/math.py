"""
Math Handler — mathematical reasoning and computation.

Handles:
  - Arithmetic: +, -, *, /, ^, %, factorial, combinations, permutations
  - Equations: linear (ax+b=c), quadratic (ax²+bx+c=0), systems
  - Word problems: translate English to math, solve, translate back
  - Verification: check if a given answer is correct
  - Number theory: primes, factors, GCD, LCM
  - Sequences: arithmetic, geometric, fibonacci, custom
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class MathResult:
    """Structured result from a math handler."""
    answer: str
    confidence: float
    method: str
    steps: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class MathHandler:
    """Handles mathematical reasoning tasks."""

    def process(self, query: str, evidence: list[str] | None = None) -> MathResult:
        t0 = time.perf_counter()
        q = query.strip()
        ev = evidence or []

        result = self._try_minmax(q, t0)
        if result:
            return result

        result = self._try_arithmetic(q, t0)
        if result:
            return result

        result = self._try_linear_equation(q, t0)
        if result:
            return result

        result = self._try_quadratic(q, t0)
        if result:
            return result

        result = self._try_number_theory(q, t0)
        if result:
            return result

        result = self._try_word_problem(q, ev, t0)
        if result:
            return result

        result = self._try_verification(q, ev, t0)
        if result:
            return result

        result = self._try_unit_conversion(q, t0)
        if result:
            return result

        return MathResult(
            answer="", confidence=0.0, method="none",
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    # ── Arithmetic ───────────────────────────────────────

    def _try_arithmetic(self, q: str, t0: float) -> MathResult | None:
        """Evaluate arithmetic expressions."""
        # Clean the query
        q_clean = q.lower().strip()
        q_clean = re.sub(r"what\s+is\s+|calculate\s+|compute\s+|find\s+", "", q_clean)
        q_clean = q_clean.rstrip("?").strip()

        # Replace word operators
        replacements = {
            "plus": "+", "minus": "-", "times": "*", "multiplied by": "*",
            "divided by": "/", "over": "/", "modulo": "%", "mod": "%",
            "to the power of": "**", "raised to": "**", "squared": "**2",
            "cubed": "**3",
        }
        for word, sym in replacements.items():
            q_clean = q_clean.replace(word, sym)

        # Modulo: "the remainder of A divided by B" -> A %% B
        rem = re.search(r"remainder\s+of\s+(\d+)\s*/\s*(\d+)", q_clean)
        if rem:
            a, b = int(rem.group(1)), int(rem.group(2))
            if b != 0:
                result = a % b
                return MathResult(
                    answer=str(result), confidence=0.99,
                    method="modulo",
                    steps=[f"remainder of {a} / {b} = {result}"],
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )

        # Factorial
        fact_match = re.match(r"(\d+)!", q_clean)
        if fact_match:
            n = int(fact_match.group(1))
            if n <= 20:
                result = math.factorial(n)
                return MathResult(
                    answer=str(result), confidence=0.99,
                    method="factorial",
                    steps=[f"{n}! = {result}"],
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )

        # sqrt
        sqrt_match = re.match(r"sqrt\s*(?:of\s*)?(\d+(?:\.\d+)?)", q_clean)
        if sqrt_match:
            n = float(sqrt_match.group(1))
            result = math.sqrt(n)
            return MathResult(
                answer=str(result), confidence=0.99,
                method="sqrt",
                steps=[f"√{n} = {result}"],
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        # General expression evaluation (safe: only numbers and operators)
        if re.match(r"^[\d\s\+\-\*/\.\(\)\%\*]+$", q_clean):
            try:
                # Limit to prevent abuse
                if len(q_clean) < 100:
                    result = eval(q_clean, {"__builtins__": {}}, {"math": math})
                    if isinstance(result, float) and result == int(result):
                        result = int(result)
                    steps = [f"{q_clean} = {result}"]
                    return MathResult(
                        answer=str(result), confidence=0.99,
                        method="arithmetic",
                        steps=steps,
                        latency_ms=(time.perf_counter() - t0) * 1000,
                    )
            except Exception:
                pass

        return None

    # ── Min / Max ────────────────────────────────────────

    def _try_minmax(self, q: str, t0: float) -> MathResult | None:
        """Handle 'largest/smallest of X, Y and Z' style queries."""
        q_clean = q.lower().strip().rstrip("?")
        m = re.search(
            r"\b(largest|greatest|biggest|maximum|highest|"
            r"smallest|least|minimum|lowest)\s+of\s+(.+?)\s*$",
            q_clean,
        )
        if not m:
            return None
        # capture everything after 'of' (numbers may be joined by
        # ', ' and ' and ') and extract all integer tokens
        nums = [int(x) for x in re.findall(r"\d+", m.group(2))]
        if not nums:
            return None
        pick_max = m.group(1) in ("largest", "greatest", "biggest",
                                  "maximum", "highest")
        result = max(nums) if pick_max else min(nums)
        label = "max" if pick_max else "min"
        return MathResult(
            answer=str(result), confidence=0.99,
            method=f"minmax_{label}",
            steps=[f"{'max' if pick_max else 'min'}({', '.join(map(str, nums))}) = {result}"],
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    # ── Linear Equations ─────────────────────────────────

    def _try_linear_equation(self, q: str, t0: float) -> MathResult | None:
        """Solve linear equations: ax + b = c."""
        # Pattern: "2x + 3 = 7" or "2*x + 3 = 7" or "2x+6=12"
        q_clean = q.lower().strip().rstrip("?").replace(" ", "")
        match = re.match(
            r"(-?\d*\.?\d*)\*?x\s*([+-]\d*\.?\d+)?\s*=\s*(-?\d*\.?\d+)",
            q_clean,
        )
        if match:
            a = float(match.group(1)) if match.group(1) else 1.0
            b = float(match.group(2).replace(" ", "")) if match.group(2) else 0.0
            c = float(match.group(3))
            if a == 0:
                return MathResult(
                    answer="no solution (a=0)" if b != c else "infinite solutions",
                    confidence=0.95, method="linear_equation",
                    steps=[f"0·x + {b} = {c}"],
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )
            x = (c - b) / a
            if x == int(x):
                x = int(x)
            chain = [
                f"{a}x + {b} = {c}",
                f"{a}x = {c} - {b} = {c - b}",
                f"x = {c - b} / {a} = {x}",
            ]
            return MathResult(
                answer=f"x = {x}", confidence=0.95,
                method="linear_equation",
                steps=chain,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        # "solve for x: 3x = 12" or "3x=12"
        solve_match = re.search(r"(-?\d*\.?\d*)\*?x\s*=\s*(-?\d*\.?\d+)", q_clean)
        if solve_match:
            a = float(solve_match.group(1)) if solve_match.group(1) else 1.0
            c = float(solve_match.group(2))
            if a != 0:
                x = c / a
                if x == int(x):
                    x = int(x)
                return MathResult(
                    answer=f"x = {x}", confidence=0.95,
                    method="linear_equation",
                    steps=[f"{a}x = {c}", f"x = {c}/{a} = {x}"],
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )

        return None

    # ── Quadratic Equations ──────────────────────────────

    def _try_quadratic(self, q: str, t0: float) -> MathResult | None:
        """Solve quadratic equations: ax² + bx + c = 0."""
        # Pattern: "x^2 + 5x + 6 = 0" or "x² + 5x + 6 = 0"
        match = re.search(
            r"(-?\d*\.?\d*)\s*\*?\s*x\s*[\^²]\s*2?\s*([+-]\s*\d*\.?\d*)\s*\*?\s*x\s*([+-]\s*\d*\.?\d+)?\s*=\s*0",
            q.lower().strip(),
        )
        if match:
            a = float(match.group(1)) if match.group(1) and match.group(1) != "" else 1.0
            b = float(match.group(2).replace(" ", "")) if match.group(2) else 0.0
            c = float(match.group(3).replace(" ", "")) if match.group(3) else 0.0

            if a == 0:
                # Degenerate to linear
                if b != 0:
                    x = -c / b
                    return MathResult(
                        answer=f"x = {x}", confidence=0.95,
                        method="quadratic",
                        steps=[f"Degenerate: {b}x + {c} = 0", f"x = {x}"],
                        latency_ms=(time.perf_counter() - t0) * 1000,
                    )
                return None

            disc = b**2 - 4*a*c
            chain = [f"{a}x² + {b}x + {c} = 0", f"Δ = {b}² - 4·{a}·{c} = {disc}"]

            if disc > 0:
                x1 = (-b + math.sqrt(disc)) / (2*a)
                x2 = (-b - math.sqrt(disc)) / (2*a)
                chain.extend([f"x₁ = {x1}", f"x₂ = {x2}"])
                return MathResult(
                    answer=f"x = {x1}, x = {x2}", confidence=0.95,
                    method="quadratic", steps=chain,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )
            elif disc == 0:
                x = -b / (2*a)
                chain.append(f"x = {x} (double root)")
                return MathResult(
                    answer=f"x = {x}", confidence=0.95,
                    method="quadratic", steps=chain,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )
            else:
                real = -b / (2*a)
                imag = math.sqrt(-disc) / (2*a)
                chain.extend([f"x₁ = {real} + {imag}i", f"x₂ = {real} - {imag}i"])
                return MathResult(
                    answer=f"x = {real} ± {imag}i", confidence=0.95,
                    method="quadratic", steps=chain,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )

        return None

    # ── Number Theory ────────────────────────────────────

    def _try_number_theory(self, q: str, t0: float) -> MathResult | None:
        """Handle prime checks, factors, GCD, LCM."""
        q_lower = q.lower().strip()

        # "Is N prime?"
        prime_match = re.match(r"is\s+(\d+)\s+prime\??", q_lower)
        if prime_match:
            n = int(prime_match.group(1))
            is_p = self._is_prime(n)
            return MathResult(
                answer="yes" if is_p else "no",
                confidence=0.99, method="prime_check",
                steps=[f"Check if {n} is prime: {'yes' if is_p else 'no'}"],
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        # "What are the factors of N?"
        factors_match = re.match(r"(?:what\s+are\s+)?(?:the\s+)?factors?\s+(?:of\s+)?(\d+)", q_lower)
        if factors_match:
            n = int(factors_match.group(1))
            factors = self._get_factors(n)
            return MathResult(
                answer=str(factors), confidence=0.99, method="factors",
                steps=[f"Factors of {n}: {factors}"],
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        # "What is the GCD of A and B?"
        gcd_match = re.match(r"(?:what\s+is\s+)?gcd\s+of\s+(\d+)\s+and\s+(\d+)", q_lower)
        if gcd_match:
            a, b = int(gcd_match.group(1)), int(gcd_match.group(2))
            g = math.gcd(a, b)
            return MathResult(
                answer=str(g), confidence=0.99, method="gcd",
                steps=[f"gcd({a}, {b}) = {g}"],
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        # "What is the LCM of A and B?"
        lcm_match = re.match(r"(?:what\s+is\s+)?lcm\s+of\s+(\d+)\s+and\s+(\d+)", q_lower)
        if lcm_match:
            a, b = int(lcm_match.group(1)), int(lcm_match.group(2))
            l = abs(a * b) // math.gcd(a, b)
            return MathResult(
                answer=str(l), confidence=0.99, method="lcm",
                steps=[f"lcm({a}, {b}) = {l}"],
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        # "What is the Nth prime?"
        nth_prime = re.match(r"(?:what\s+is\s+)?(?:the\s+)?(\w+)\s+prime", q_lower)
        if nth_prime:
            word = nth_prime.group(1)
            ordinal_map = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
                           "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
                           "11th": 11, "12th": 12, "13th": 13, "14th": 14, "15th": 15,
                           "16th": 16, "17th": 17, "18th": 18, "19th": 19, "20th": 20}
            if word in ordinal_map:
                n = ordinal_map[word]
                count = 0
                candidate = 1
                while count < n:
                    candidate += 1
                    if self._is_prime(candidate):
                        count += 1
                return MathResult(
                    answer=str(candidate), confidence=0.99, method="nth_prime",
                    steps=[f"The {word} prime is {candidate}"],
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )

        return None

    # ── Word Problems ────────────────────────────────────

    def _try_word_problem(self, q: str, ev: list[str], t0: float) -> MathResult | None:
        """Translate English word problems into math and solve."""
        q_lower = q.lower().strip()

        # "If I have X and add Y, how many do I have?"
        add_match = re.search(
            r"if\s+i\s+have\s+(\d+)\s+(?:and\s+)?(?:add|plus|gain)\s+(\d+).*how\s+many",
            q_lower,
        )
        if add_match:
            a, b = int(add_match.group(1)), int(add_match.group(2))
            return MathResult(
                answer=str(a + b), confidence=0.90, method="word_problem",
                steps=[f"Have {a}, add {b}", f"{a} + {b} = {a + b}"],
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        # "If I have X and lose/give away Y, how many left?"
        sub_match = re.search(
            r"if\s+i\s+have\s+(\d+)\s+.*(?:lose|give\s+away|subtract)\s+(\d+).*how\s+many\s+(?:left|remain)",
            q_lower,
        )
        if sub_match:
            a, b = int(sub_match.group(1)), int(sub_match.group(2))
            return MathResult(
                answer=str(a - b), confidence=0.90, method="word_problem",
                steps=[f"Have {a}, lose {b}", f"{a} - {b} = {a - b}"],
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        # "X items cost $Y each. How much for Z items?"
        cost_match = re.search(
            r"(\d+)\s+items?\s+cost\s+\$?(\d+(?:\.\d+)?)\s+each.*(?:how\s+much|total|cost)\s*(?:for\s+)?(\d+)",
            q_lower,
        )
        if cost_match:
            qty, price, ask_qty = int(cost_match.group(1)), float(cost_match.group(2)), int(cost_match.group(3))
            total = price * ask_qty
            return MathResult(
                answer=f"${total:.2f}", confidence=0.90, method="word_problem",
                steps=[f"{qty} items at ${price} each", f"Total for {ask_qty}: ${total:.2f}"],
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        # "What is X percent of Y?" or "X% of Y"
        pct_match = re.search(r"(?:what\s+is\s+)?(\d+(?:\.\d+)?)\s*%\s*(?:of\s+)?(\d+(?:\.\d+)?)", q_lower)
        if pct_match:
            pct, val = float(pct_match.group(1)), float(pct_match.group(2))
            result = (pct / 100) * val
            return MathResult(
                answer=str(result), confidence=0.95, method="percentage",
                steps=[f"{pct}% of {val} = ({pct}/100) × {val} = {result}"],
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        # "What is X as a percentage of Y?"
        as_pct = re.search(r"(\d+(?:\.\d+)?)\s+(?:is\s+)?(?:what\s+)?percent(?:age)?\s+of\s+(\d+(?:\.\d+)?)", q_lower)
        if as_pct:
            part, whole = float(as_pct.group(1)), float(as_pct.group(2))
            if whole != 0:
                result = (part / whole) * 100
                return MathResult(
                    answer=f"{result:.2f}%", confidence=0.95, method="percentage",
                    steps=[f"{part} / {whole} × 100 = {result:.2f}%"],
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )

        return None

    # ── Verification ─────────────────────────────────────

    def _try_verification(self, q: str, ev: list[str], t0: float) -> MathResult | None:
        """Verify if a mathematical claim is correct."""
        q_lower = q.lower().strip()

        # "Is X + Y = Z correct?"
        verify = re.search(
            r"is\s+(.+?)\s*=\s*(.+?)\s*(?:correct|right|true)\??",
            q_lower,
        )
        if verify:
            expr_str = verify.group(1).strip()
            claimed = verify.group(2).strip()
            # Try to evaluate the expression
            expr_clean = expr_str.replace("×", "*").replace("÷", "/")
            if re.match(r"^[\d\s\+\-\*/\.\(\)]+$", expr_clean):
                try:
                    actual = eval(expr_clean, {"__builtins__": {}}, {})
                    claimed_val = float(claimed)
                    correct = abs(actual - claimed_val) < 0.001
                    return MathResult(
                        answer="correct" if correct else f"incorrect, actual = {actual}",
                        confidence=0.95, method="verification",
                        steps=[f"{expr_str} = {actual}", f"Claimed: {claimed}", f"{'Correct' if correct else 'Incorrect'}"],
                        latency_ms=(time.perf_counter() - t0) * 1000,
                    )
                except Exception:
                    pass

        return None

    # ── Unit Conversion ──────────────────────────────────

    def _try_unit_conversion(self, q: str, t0: float) -> MathResult | None:
        """Handle unit conversions."""
        q_lower = q.lower().strip()

        conversions = {
            # Plural forms
            ("miles", "kilometers"): 1.60934,
            ("kilometers", "miles"): 0.621371,
            ("pounds", "kilograms"): 0.453592,
            ("kilograms", "pounds"): 2.20462,
            ("feet", "meters"): 0.3048,
            ("meters", "feet"): 3.28084,
            ("inches", "centimeters"): 2.54,
            ("centimeters", "inches"): 0.393701,
            ("fahrenheit", "celsius"): None,  # Special
            ("celsius", "fahrenheit"): None,  # Special
            # Singular forms (after rstrip)
            ("mile", "kilometer"): 1.60934,
            ("kilometer", "mile"): 0.621371,
            ("pound", "kilogram"): 0.453592,
            ("kilogram", "pound"): 2.20462,
            ("foot", "meter"): 0.3048,
            ("meter", "foot"): 3.28084,
            ("inch", "centimeter"): 2.54,
            ("centimeter", "inch"): 0.393701,
            ("fahrenheit", "celsius"): None,
            ("celsius", "fahrenheit"): None,
        }

        # "Convert X miles to kilometers"
        conv_match = re.search(
            r"(?:convert\s+)?(\d+(?:\.\d+)?)\s+(\w+)\s+(?:to|into)\s+(\w+)",
            q_lower,
        )
        if conv_match:
            val = float(conv_match.group(1))
            from_unit = conv_match.group(2).rstrip("s")
            to_unit = conv_match.group(3).rstrip("s")

            key = (from_unit, to_unit)
            if key in conversions:
                factor = conversions[key]
                if from_unit == "fahrenheit" and to_unit == "celsius":
                    result = (val - 32) * 5/9
                    return MathResult(
                        answer=f"{result:.2f}°C", confidence=0.99,
                        method="unit_conversion",
                        steps=[f"{val}°F = ({val} - 32) × 5/9 = {result:.2f}°C"],
                        latency_ms=(time.perf_counter() - t0) * 1000,
                    )
                elif from_unit == "celsius" and to_unit == "fahrenheit":
                    result = val * 9/5 + 32
                    return MathResult(
                        answer=f"{result:.2f}°F", confidence=0.99,
                        method="unit_conversion",
                        steps=[f"{val}°C = {val} × 9/5 + 32 = {result:.2f}°F"],
                        latency_ms=(time.perf_counter() - t0) * 1000,
                    )
                elif factor is not None:
                    result = val * factor
                    return MathResult(
                        answer=f"{result:.4f} {to_unit}s", confidence=0.99,
                        method="unit_conversion",
                        steps=[f"{val} {from_unit}s × {factor} = {result:.4f} {to_unit}s"],
                        latency_ms=(time.perf_counter() - t0) * 1000,
                    )

        return None

    # ── Helpers ──────────────────────────────────────────

    @staticmethod
    def _is_prime(n: int) -> bool:
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    @staticmethod
    def _get_factors(n: int) -> list[int]:
        if n < 1:
            return []
        factors = []
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                factors.append(i)
                if i != n // i:
                    factors.append(n // i)
        return sorted(factors)
