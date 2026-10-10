"""Offline tests for the Aegis pipeline (no model download required).

A FakeRouter dispatches canned answers by keyword, letting us verify the
control flow: quarantine short-circuits, PII redaction, billing-only policy
stage, VIP escalation, dynamic action options, audit completeness, and
per-stage batching.
"""
import json
import unittest

from app.pipeline import (
    CHURN_ESCALATION_GATE,
    DEFAULT_STAGES,
    result_row,
    run_pipeline,
)
from app.questions import build_action_questions
from app.redact import redact_text

TICKETS = [
    {"id": "spam", "text": "CONGRATULATIONS you won our lottery, claim now"},
    {"id": "abuse", "text": "your team are absolute idiots, fix this garbage"},
    {"id": "pii", "text": "refund card 4111-1111-1111-1111, email john@example.com"},
    {"id": "billing", "text": "we were billed twice for March, please refund today"},
    {"id": "billing_churn",
     "text": "billed twice, refund the duplicate or we cancel our enterprise plan"},
    {"id": "technical", "text": "the dashboard shows a 500 error on export, blocking"},
    {"id": "sales", "text": "do you offer volume pricing for 50+ seats next quarter"},
    {"id": "hindi_billing", "text": "मुझसे मार्च में दो बार शुल्क लिया गया, वापस करें"},
]

_DUP_MARKERS = ("twice", "दो बार", "zweimal")


class FakeRouter:
    def __init__(self):
        self.calls = []  # one entry per predict_batch call

    def predict_batch(self, requests, **kwargs):
        self.calls.append([r["state"]["body"] for r in requests])
        return [self._answer(r) for r in requests]

    @staticmethod
    def _answer(req):
        text = req["state"]["body"]
        questions = req["questions"]
        dup = any(m in text for m in _DUP_MARKERS)
        vip = "cancel our enterprise" in text

        if "is_spam" in questions:
            return {"answers": {
                "is_spam": {"noul": 0.97 if "lottery" in text else 0.02},
                "is_abuse": {"noul": 0.95 if "idiots" in text else 0.03},
                "contains_pii": {"noul": 0.92 if "4111" in text or "@" in text else 0.04},
            }, "routing": {"model": "english", "reason": "fake guard"}}
        if "department" in questions:
            if "500 error" in text:
                dept = "technical"
            elif "seats" in text:
                dept = "sales"
            else:
                dept = "billing"
            return {"answers": {
                "department": {"choice": dept, "answer_confidence": 0.9},
                "urgency": {"score": 2.0 if ("blocking" in text or vip) else 1.0,
                            "answer_confidence": 0.88},
                "refund_requested": {"noul": 0.9 if dup else 0.1,
                                     "answer_confidence": 0.87},
                "churn_risk": {"noul": 0.95 if vip else 0.05,
                               "answer_confidence": 0.9},
            }, "routing": {"model": "english", "reason": "fake triage"}}
        if "refund_policy_window" in questions:
            return {"answers": {
                "refund_policy_window": {"noul": 0.9 if dup else 0.2},
                "duplicate_charge_evidence": {"noul": 0.93 if dup else 0.1},
                "entitlement": {"noul": 0.95},
            }, "routing": {"model": "typed-decisions", "reason": "fake policy"}}
        if "action" in questions:
            options = list(questions["action"]["criteria"])
            if vip and "vip_escalation" in options:
                action = "vip_escalation"
            elif dup and "auto_refund" in options:
                action = "auto_refund"
            else:
                action = "standard_queue"
            return {"answers": {
                "action": {"choice": action, "answer_confidence": 0.91},
            }, "routing": {"model": "typed-decisions", "reason": "fake action"}}
        raise AssertionError(f"unknown question set: {list(questions)}")


class TestRedaction(unittest.TestCase):
    def test_redacts_card_email_phone(self):
        text = "card 4111-1111-1111-1111, mail john@example.com, call +1 555 123 4567"
        out = redact_text(text)
        self.assertNotIn("4111", out)
        self.assertNotIn("john@example.com", out)
        self.assertNotIn("555 123 4567", out)
        self.assertIn("[REDACTED_CARD]", out)
        self.assertIn("[REDACTED_EMAIL]", out)

    def test_leaves_plain_text_alone(self):
        text = "the dashboard shows a 500 error on export"
        self.assertEqual(redact_text(text), text)


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.router = FakeRouter()
        cls.ctxs = run_pipeline(cls.router, TICKETS)
        cls.by_id = {c.id: c for c in cls.ctxs}

    def test_spam_quarantined_immediately(self):
        ctx = self.by_id["spam"]
        self.assertEqual(ctx.route, "quarantine")
        self.assertEqual([a["stage"] for a in ctx.audit], ["guard"])

    def test_abuse_quarantined_immediately(self):
        ctx = self.by_id["abuse"]
        self.assertEqual(ctx.route, "quarantine")
        self.assertEqual(len(ctx.audit), 1)

    def test_pii_is_redacted_and_pipeline_continues(self):
        ctx = self.by_id["pii"]
        self.assertTrue(ctx.redacted)
        self.assertNotIn("4111", ctx.state["body"])
        self.assertNotIn("john@example.com", ctx.state["body"])
        self.assertNotIn("quarantine", ctx.route)
        self.assertIn("action", [a["stage"] for a in ctx.audit])

    def test_billing_ticket_gets_policy_and_auto_refund(self):
        ctx = self.by_id["billing"]
        stages = [a["stage"] for a in ctx.audit]
        self.assertEqual(stages, ["guard", "triage", "policy", "action"])
        self.assertEqual(ctx.route, "auto_refund")

    def test_policy_stage_skipped_for_non_billing(self):
        ctx = self.by_id["technical"]
        stages = [a["stage"] for a in ctx.audit]
        self.assertNotIn("policy", stages)
        self.assertEqual(ctx.route, "standard_queue")

    def test_churn_gate_escalates_before_action(self):
        ctx = self.by_id["billing_churn"]
        self.assertEqual(ctx.route, "vip_escalation")
        self.assertNotIn("action", [a["stage"] for a in ctx.audit])

    def test_churn_gate_uses_configured_threshold(self):
        ctx = self.by_id["billing_churn"]
        self.assertGreaterEqual(ctx.answers["churn_risk"]["noul"],
                                CHURN_ESCALATION_GATE)

    def test_non_latin_routing_is_transparent_in_audit(self):
        ctx = self.by_id["hindi_billing"]
        self.assertEqual(ctx.route, "auto_refund")
        self.assertTrue(any(a["reason"] for a in ctx.audit))

    def test_action_options_are_composed_from_findings(self):
        billing = self.by_id["billing"]
        with_refund = build_action_questions(billing)["action"]["criteria"]
        self.assertIn("auto_refund", with_refund)
        technical = self.by_id["technical"]
        without_refund = build_action_questions(technical)["action"]["criteria"]
        self.assertNotIn("auto_refund", without_refund)

    def test_audit_entries_are_complete_and_serializable(self):
        for ctx in self.ctxs:
            for entry in ctx.audit:
                self.assertIn(entry["stage"],
                              {"guard", "triage", "policy", "action"})
                self.assertIn("model", entry)
                self.assertGreaterEqual(entry["latency_ms"], 0.0)
                for a in entry["answers"].values():
                    self.assertIn(a["type"], {"choice", "score", "noul"})
                    self.assertIsNotNone(a["value"])
            json.dumps(ctx.audit)  # must not raise

    def test_batched_by_stage(self):
        # Cohort shrinks as gates fire: 8 enter the guard; spam+abuse are
        # quarantined (6 reach triage); the churn ticket escalates at
        # triage and only 3 billing tickets reach policy; 5 reach action.
        self.assertEqual([len(call) for call in self.router.calls], [8, 6, 3, 5])
        self.assertEqual(len(self.router.calls), len(DEFAULT_STAGES))

    def test_result_rows(self):
        rows = [result_row(c) for c in self.ctxs]
        for row in rows:
            self.assertIn("route", row)
            self.assertIn("stages", row)
        json.dumps(rows)  # must not raise


if __name__ == "__main__":
    unittest.main()
