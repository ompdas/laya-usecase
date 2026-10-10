"""Multi-stage decision pipeline.

A ticket flows through a DAG of stages. Each stage is a typed-decision
batch: every ticket that reaches the stage is scored against that stage's
questions in *one* shared forward pass (`Router.predict_batch`). Stages can
short-circuit the ticket (quarantine, VIP escalation), mutate its state
(PII redaction), or feed their answers into the questions of later stages.

            ticket
              │
           [guard] ──spam/abuse──▶ quarantine
              │ pass (redact if PII)
           [triage] ──churn ≥ 0.85──▶ vip_escalation
              │
     billing? ──yes──▶ [policy] ─┐
        no ──────────────────────┤
                                 ▼
                              [action] ──▶ auto_refund / priority_human / standard_queue ...
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .questions import (
    GUARD_QUESTIONS,
    POLICY_QUESTIONS,
    TRIAGE_QUESTIONS,
    build_action_questions,
)
from .redact import redact_state

# Gate thresholds — tune on held-out data, these are illustrative defaults.
SPAM_GATE = 0.80
ABUSE_GATE = 0.80
PII_REDACT_GATE = 0.70
CHURN_ESCALATION_GATE = 0.85


@dataclass
class TicketContext:
    """Everything the pipeline knows about one ticket."""
    id: str
    text: str
    state: dict = field(init=False)
    redacted: bool = False
    route: str = "pending"
    answers: dict = field(default_factory=dict)
    audit: list[dict] = field(default_factory=list)
    done: bool = False

    def __post_init__(self):
        self.state = {"body": self.text}

    def terminal(self, route: str) -> None:
        self.route, self.done = route, True


def make_audit_entry(stage: str, result: dict, latency_ms: float) -> dict:
    """One auditable record: what was asked, what came back, why, how fast."""
    answers = {}
    for qid, a in result["answers"].items():
        if "choice" in a:
            value, qtype = a["choice"], "choice"
        elif "score" in a:
            value, qtype = a["score"], "score"
        else:
            value, qtype = a["noul"], "noul"
        answers[qid] = {
            "type": qtype,
            "value": value,
            "confidence": a.get("answer_confidence", a.get("confidence")),
        }
    return {
        "stage": stage,
        "model": result["routing"]["model"],
        "reason": result["routing"].get("reason"),
        "latency_ms": round(latency_ms, 2),
        "answers": answers,
    }


class Stage:
    """Base class: `predicate` filters which tickets run the stage,
    `questions_for` builds the batch question set, `apply` consumes results
    and makes gating decisions."""

    name = "stage"

    def predicate(self, ctx: TicketContext) -> bool:
        return True

    def questions_for(self, ctx: TicketContext) -> dict:
        raise NotImplementedError

    def apply(self, ctx: TicketContext, result: dict, latency_ms: float) -> None:
        raise NotImplementedError


class GuardStage(Stage):
    name = "guard"

    def questions_for(self, ctx):
        return GUARD_QUESTIONS

    def apply(self, ctx, result, latency_ms):
        answers = result["answers"]
        ctx.answers.update(answers)
        ctx.audit.append(make_audit_entry(self.name, result, latency_ms))
        if answers["is_spam"]["noul"] >= SPAM_GATE or answers["is_abuse"]["noul"] >= ABUSE_GATE:
            ctx.terminal("quarantine")
        elif answers["contains_pii"]["noul"] >= PII_REDACT_GATE:
            ctx.state = redact_state(ctx.state)
            ctx.redacted = True


class TriageStage(Stage):
    name = "triage"

    def questions_for(self, ctx):
        return TRIAGE_QUESTIONS

    def apply(self, ctx, result, latency_ms):
        answers = result["answers"]
        ctx.answers.update(answers)
        ctx.audit.append(make_audit_entry(self.name, result, latency_ms))
        if answers["churn_risk"]["noul"] >= CHURN_ESCALATION_GATE:
            ctx.terminal("vip_escalation")


class PolicyStage(Stage):
    name = "policy"

    def predicate(self, ctx):
        return ctx.answers.get("department", {}).get("choice") == "billing"

    def questions_for(self, ctx):
        return POLICY_QUESTIONS

    def apply(self, ctx, result, latency_ms):
        ctx.answers.update(result["answers"])
        ctx.audit.append(make_audit_entry(self.name, result, latency_ms))


class ActionStage(Stage):
    name = "action"

    def questions_for(self, ctx):
        return build_action_questions(ctx)

    def apply(self, ctx, result, latency_ms):
        ctx.answers.update(result["answers"])
        ctx.audit.append(make_audit_entry(self.name, result, latency_ms))
        ctx.terminal(result["answers"]["action"]["choice"])


DEFAULT_STAGES = (GuardStage(), TriageStage(), PolicyStage(), ActionStage())


def run_pipeline(router, tickets: list[dict], stages=DEFAULT_STAGES) -> list[TicketContext]:
    """Run all tickets through the DAG.

    Per stage, the tickets that are still active *and* match the stage
    predicate are batched into a single `predict_batch` call — one forward
    pass per stage per ticket, shared across the cohort.
    """
    ctxs = [TicketContext(t["id"], t["text"]) for t in tickets]
    for stage in stages:
        active = [c for c in ctxs if not c.done and stage.predicate(c)]
        if not active:
            continue
        requests = [{"state": c.state, "questions": stage.questions_for(c)} for c in active]
        t0 = time.perf_counter()
        results = router.predict_batch(requests)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        per_ticket_ms = elapsed_ms / max(len(active), 1)
        for ctx, result in zip(active, results):
            stage.apply(ctx, result, per_ticket_ms)
    return ctxs


def result_row(ctx: TicketContext) -> dict[str, Any]:
    """Flatten a finished ticket into a dashboard/CSV-friendly row."""
    def ans(qid, key, default=None):
        return ctx.answers.get(qid, {}).get(key, default)

    confidences = [
        a.get("confidence") for a in ctx.audit[-1]["answers"].values()
        if a.get("confidence") is not None
    ] if ctx.audit else []
    return {
        "id": ctx.id,
        "route": ctx.route,
        "department": ans("department", "choice", "-"),
        "urgency": ans("urgency", "score", "-"),
        "churn_probability": round(ans("churn_risk", "noul", 0.0) or 0.0, 3),
        "refund_probability": round(ans("refund_requested", "noul", 0.0) or 0.0, 3),
        "redacted": ctx.redacted,
        "stages": [a["stage"] for a in ctx.audit],
        "min_confidence": round(min(confidences), 3) if confidences else None,
    }
