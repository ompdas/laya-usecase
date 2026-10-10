"""Judge protocol shared by every grader backend.

A Judge turns a list of eval items into a list of Grades. Items are dicts:

    {"id": ..., "query": str, "context": str, "response": str}

`grade_with_cascade` implements the hybrid pattern: a cheap first-pass judge
grades everything, and only the items it flags for review are escalated to a
stronger (usually LLM) judge.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Grade:
    item_id: str
    judge: str
    verdict: str            # "pass" | "fail" | "review"
    goal_achieved: float    # P(true)
    faithful: float         # P(true)
    quality: float          # expected ordinal level
    failure_mode: str
    min_confidence: float | None
    latency_ms: float
    detail: dict = field(default_factory=dict)


class Judge(Protocol):
    name: str

    def grade(self, items: list[dict]) -> list[Grade]: ...


def grade_with_cascade(first_pass: Judge, escalated: Judge, items: list[dict],
                       review_verdict: str = "review") -> list[Grade]:
    """Grade everything with `first_pass`; re-grade its `review` items with
    the stronger `escalated` judge. Review items keep both grades' evidence
    in `detail` so the escalation itself is auditable."""
    t0 = time.perf_counter()
    grades = first_pass.grade(items)
    flagged = [g for g in grades if g.verdict == review_verdict]
    if flagged:
        by_id = {i["id"]: i for i in items}
        second = escalated.grade([by_id[g.item_id] for g in flagged])
        overrides = {g.item_id: g for g in second}
        merged = []
        for g in grades:
            if g.item_id in overrides:
                o = overrides[g.item_id]
                o.detail["first_pass"] = {
                    "judge": g.judge, "verdict": g.verdict,
                    "min_confidence": g.min_confidence,
                }
                merged.append(o)
            else:
                merged.append(g)
        grades = merged
    for g in grades:
        g.detail["cascade_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return grades
