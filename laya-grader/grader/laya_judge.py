"""LayaJudge: grade eval items with typed decisions in batched forward passes."""
from __future__ import annotations

import time

from .judge import Grade
from .questions import (
    FAITHFUL_GATE,
    GOAL_GATE,
    QA_GRADING_QUESTIONS,
    REVIEW_CONFIDENCE,
)


class LayaJudge:
    """Drop-in replacement for LLM-as-judge on structured grading tasks.

    Usage:
        from laya import Router
        judge = LayaJudge(Router(preload=True))
        grades = judge.grade(items)
    """

    name = "laya"

    def __init__(self, router, questions: dict | None = None):
        self.router = router
        self.questions = questions or QA_GRADING_QUESTIONS

    @staticmethod
    def _confidence(answer: dict) -> float:
        return float(answer.get("answer_confidence") or answer.get("confidence") or 0.0)

    def _verdict(self, answers: dict) -> str:
        confs = [self._confidence(a) for a in answers.values()]
        if confs and min(confs) < REVIEW_CONFIDENCE:
            return "review"
        if (answers["goal_achieved"]["noul"] >= GOAL_GATE
                and answers["faithful_to_context"]["noul"] >= FAITHFUL_GATE
                and answers["failure_mode"]["choice"] == "none"):
            return "pass"
        return "fail"

    def grade(self, items: list[dict]) -> list[Grade]:
        requests = [
            {"state": {"query": i["query"], "context": i["context"],
                       "response": i["response"]},
             "questions": self.questions}
            for i in items
        ]
        t0 = time.perf_counter()
        results = self.router.predict_batch(requests, sort_by_length=True)
        total_ms = (time.perf_counter() - t0) * 1000

        grades = []
        for item, result in zip(items, results):
            a = result["answers"]
            confs = [self._confidence(x) for x in a.values()]
            grades.append(Grade(
                item_id=item["id"],
                judge=self.name,
                verdict=self._verdict(a),
                goal_achieved=round(float(a["goal_achieved"]["noul"]), 3),
                faithful=round(float(a["faithful_to_context"]["noul"]), 3),
                quality=round(float(a["quality"]["score"]), 2),
                failure_mode=a["failure_mode"]["choice"],
                min_confidence=round(min(confs), 3) if confs else None,
                latency_ms=round(total_ms / max(len(items), 1), 2),
                detail={"model": result["routing"]["model"],
                        "routing_reason": result["routing"].get("reason")},
            ))
        return grades
