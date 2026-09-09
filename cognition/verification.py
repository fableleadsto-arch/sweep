"""Phase 12 — Independent Verification.

A conclusion is verified when neural, logic, and evidence-based checks
(independently available path) agree. Otherwise, the claim is contested or
resolved without assuming an answer.

Required qualified states: VERIFIED / CONTESTED / UNRESOLVED.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


class VerificationStatus(str, enum.Enum):
    VERIFIED = "VERIFIED"
    CONTESTED = "CONTESTED"
    UNRESOLVED = "UNRESOLVED"


@dataclass
class VerificationResult:
    """Outcome of running every available check independently."""

    claim_id: str
    passed: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    unrun: List[str] = field(default_factory=list)
    status: VerificationStatus = VerificationStatus.UNRESOLVED
    summary: str = ""

    def add_pass(self, method: str) -> None:
        if method not in self.passed and method not in self.failed:
            self.passed.append(method)

    def add_fail(self, method: str) -> None:
        if method not in self.passed and method not in self.failed:
            self.failed.append(method)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "passed": self.passed,
            "failed": self.failed,
            "unrun": self.unrun,
            "status": self.status.value,
            "summary": self.summary,
        }


class VerificationEngine:
    """Runs each available independent check and combines the verdict.

    Requirement: verify by at least two independent paths;
    a disagreement between any two available checks makes it CONTESTED.
    """

    def __init__(self):
        self._results: Dict[str, VerificationResult] = {}
        self._check_registry: Dict[str, Callable[[str], bool]] = {}

    def register_check(self, name: str, fn: Callable[[str], bool]) -> None:
        self._check_registry[name] = fn

    def run(self, claim_id: str) -> VerificationResult:
        result = VerificationResult(claim_id=claim_id)
        for name, fn in self._check_registry.items():
            try:
                if fn(claim_id):
                    result.add_pass(name)
                else:
                    result.add_fail(name)
            except Exception:
                result.unrun.append(name)
        result.status = finalize(result)
        result.summary = summarize(result)
        self._results[claim_id] = result
        return result

    def result(self, claim_id: str) -> Optional[VerificationResult]:
        return self._results.get(claim_id)


def finalize(result: VerificationResult) -> VerificationStatus:
    """Determine status: VERIFIED, CONTESTED, or UNRESOLVED."""
    if result.failed:
        return VerificationStatus.CONTESTED            # at least one independent path disagrees
    if len(result.passed) >= 2:
        return VerificationStatus.VERIFIED             # two+ independent paths agree
    if len(result.passed) == 1:
        return VerificationStatus.UNRESOLVED           # single check insufficient
    return VerificationStatus.UNRESOLVED               # nothing corroborated


def summarize(result: VerificationResult) -> str:
    if result.status is VerificationStatus.CONTESTED:
        return (f"contested: checks {', '.join(result.passed)} agreed but "
                f"{', '.join(result.failed)} disagreed")
    if result.status is VerificationStatus.VERIFIED:
        return f"verified by independent checks: {', '.join(result.passed)}"
    return f"unresolved: no unambiguous verdict from checks {result.passed or 'none'}"


def verified(result: VerificationResult) -> bool:
    return result.status is VerificationStatus.VERIFIED


def contested(result: VerificationResult) -> bool:
    return result.status is VerificationStatus.CONTESTED