"""Core triage logic: map raw tickets to typed decisions via a Laya Router."""
from __future__ import annotations

from typing import Any

from .questions import MIN_CONFIDENCE, TRIAGE_QUESTIONS


def build_requests(tickets: list[dict]) -> list[dict]:
    """One request per ticket; all requests share the same question set,
    which lets Router.predict_batch pack them into shared forward passes."""
    return [
        {"state": {"body": t["text"]}, "questions": TRIAGE_QUESTIONS}
        for t in tickets
    ]


def answer_confidence(answer: dict) -> float:
    """Pull the best calibrated confidence value out of an answer dict.

    `answer_confidence` is the probability of the reported answer on every
    question type; fall back to `confidence` (1 - normalised entropy).
    """
    return float(answer.get("answer_confidence") or answer.get("confidence") or 0.0)


def needs_human_review(answers: dict[str, dict]) -> bool:
    """Any answer the model is unsure about escalates the whole ticket."""
    return any(answer_confidence(a) < MIN_CONFIDENCE for a in answers.values())


def triage_tickets(
    router,
    tickets: list[dict],
    *,
    batch_size: int | None = None,
    sort_by_length: bool = True,
) -> list[dict]:
    """Triage a list of tickets with one batched call.

    Returns one flat row per ticket — ready for a CSV, a database, or a
    dashboard — including which checkpoint served it (`routing.model`).
    """
    requests = build_requests(tickets)
    kwargs: dict[str, Any] = {}
    if batch_size:
        kwargs["batch_size"] = batch_size
    if sort_by_length:
        kwargs["sort_by_length"] = True

    results = router.predict_batch(requests, **kwargs)

    rows = []
    for ticket, result in zip(tickets, results):
        answers = result["answers"]
        confidences = [answer_confidence(a) for a in answers.values()]
        rows.append(
            {
                "id": ticket["id"],
                "department": answers["department"]["choice"],
                "urgency": round(float(answers["urgency"]["score"]), 2),
                "refund_probability": round(float(answers["refund_requested"]["noul"]), 3),
                "churn_probability": round(float(answers["churn_risk"]["noul"]), 3),
                "min_confidence": round(min(confidences), 3),
                "model": result["routing"]["model"],
                "needs_human_review": needs_human_review(answers),
            }
        )
    return rows
