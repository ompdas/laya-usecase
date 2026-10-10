"""Offline tests for the triage logic (no model download required)."""
import unittest

from app.questions import MIN_CONFIDENCE, TRIAGE_QUESTIONS
from app.triage import (
    answer_confidence,
    build_requests,
    needs_human_review,
    triage_tickets,
)


class FakeRouter:
    """Stands in for laya.Router: echoes canned answers per ticket id."""

    def predict_batch(self, requests, **kwargs):
        out = []
        for req in requests:
            text = req["state"]["body"]
            angry = "cancel" in text
            out.append(
                {
                    "answers": {
                        "department": {"choice": "billing", "answer_confidence": 0.94},
                        "urgency": {"score": 2.0, "answer_confidence": 0.91},
                        "refund_requested": {"noul": 0.92, "answer_confidence": 0.88},
                        "churn_risk": {"noul": 0.89 if angry else 0.1,
                                       "answer_confidence": 0.9},
                    },
                    "routing": {"model": "english", "reason": "test double"},
                }
            )
        return out


class TestQuestions(unittest.TestCase):
    def test_schema_shape(self):
        self.assertEqual(set(TRIAGE_QUESTIONS), {
            "department", "urgency", "refund_requested", "churn_risk",
        })
        for q in TRIAGE_QUESTIONS.values():
            self.assertIn(q["type"], {"choice", "score", "noul"})

    def test_choice_has_criteria_dict(self):
        dept = TRIAGE_QUESTIONS["department"]
        self.assertIsInstance(dept["criteria"], dict)
        self.assertGreaterEqual(len(dept["criteria"]), 2)

    def test_score_levels_are_ordered(self):
        self.assertGreaterEqual(len(TRIAGE_QUESTIONS["urgency"]["criteria"]), 2)


class TestLogic(unittest.TestCase):
    def test_build_requests(self):
        tickets = [{"id": "A", "text": "hello"}, {"id": "B", "text": "world"}]
        reqs = build_requests(tickets)
        self.assertEqual(len(reqs), 2)
        self.assertEqual(reqs[0]["state"], {"body": "hello"})
        self.assertIs(reqs[0]["questions"], TRIAGE_QUESTIONS)

    def test_confidence_fallback(self):
        self.assertEqual(answer_confidence({"answer_confidence": 0.8}), 0.8)
        self.assertEqual(answer_confidence({"confidence": 0.7}), 0.7)
        self.assertEqual(answer_confidence({}), 0.0)

    def test_review_gate(self):
        sure = {"a": {"answer_confidence": 0.95}}
        unsure = {"a": {"answer_confidence": MIN_CONFIDENCE - 0.01}}
        self.assertFalse(needs_human_review(sure))
        self.assertTrue(needs_human_review(unsure))

    def test_triage_rows(self):
        tickets = [
            {"id": "T1", "text": "refund or we cancel"},
            {"id": "T2", "text": "quick question about pricing"},
        ]
        rows = triage_tickets(FakeRouter(), tickets)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["department"], "billing")
        self.assertGreater(rows[0]["churn_probability"], 0.5)
        self.assertLess(rows[1]["churn_probability"], 0.5)
        self.assertFalse(rows[0]["needs_human_review"])


if __name__ == "__main__":
    unittest.main()
