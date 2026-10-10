"""Offline tests for the grading harness (no model download required)."""
import unittest

from grader.judge import Grade, grade_with_cascade
from grader.laya_judge import LayaJudge


def item(id_, response, **kw):
    return {"id": id_, "query": kw.get("query", "what is the refund window?"),
            "context": kw.get("context", "refunds are allowed within 30 days."),
            "response": response}


ITEMS = [
    item("good", "refunds are allowed within 30 days."),
    item("hallucinated", "refunds are allowed within 30 days, plus a lifetime discount."),
    item("partial", "refunds are allowed.", ),
    item("offtopic", "our company was founded in 2015."),
    item("refusal", "I cannot answer that question."),
    item("lowconf", "refunds are allowed within 30 days.", query="LOWCONF"),
]


class FakeRouter:
    def __init__(self):
        self.calls = 0

    def predict_batch(self, requests, **kwargs):
        self.calls += 1
        out = []
        for req in requests:
            s = req["state"]
            resp = s["response"]
            query = s["query"]
            hallucinated = "lifetime" in resp
            offtopic = "founded" in resp
            refusal = "cannot answer" in resp
            partial = resp == "refunds are allowed."
            good = not any([hallucinated, offtopic, refusal, partial])
            lowconf = query == "LOWCONF"
            conf = 0.4 if lowconf else 0.92
            out.append({
                "answers": {
                    "goal_achieved": {"noul": 0.1 if (offtopic or refusal) else
                                      (0.5 if partial else 0.95), "answer_confidence": conf},
                    "faithful_to_context": {"noul": 0.1 if hallucinated else 0.96,
                                            "answer_confidence": conf},
                    "quality": {"score": 2.0 if good else 0.5, "answer_confidence": conf},
                    "failure_mode": {"choice": "hallucination" if hallucinated else
                                     "off_topic" if offtopic else
                                     "refusal" if refusal else
                                     "incomplete" if partial else "none",
                                     "answer_confidence": conf},
                },
                "routing": {"model": "english", "reason": "fake"},
            })
        return out


class FakeLLMJudge:
    name = "llm:fake"

    def grade(self, items):
        return [Grade(item_id=i["id"], judge=self.name, verdict="pass",
                      goal_achieved=1.0, faithful=1.0, quality=2.0,
                      failure_mode="none", min_confidence=None,
                      latency_ms=800.0) for i in items]


class TestLayaJudge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.judge = LayaJudge(FakeRouter())
        cls.grades = cls.judge.grade(ITEMS)
        cls.by_id = {g.item_id: g for g in cls.grades}

    def test_good_response_passes(self):
        g = self.by_id["good"]
        self.assertEqual(g.verdict, "pass")
        self.assertEqual(g.failure_mode, "none")
        self.assertGreaterEqual(g.goal_achieved, 0.9)

    def test_hallucination_fails_with_mode(self):
        g = self.by_id["hallucinated"]
        self.assertEqual(g.verdict, "fail")
        self.assertEqual(g.failure_mode, "hallucination")
        self.assertLess(g.faithful, 0.5)

    def test_partial_and_offtopic_and_refusal_fail(self):
        self.assertEqual(self.by_id["partial"].failure_mode, "incomplete")
        self.assertEqual(self.by_id["offtopic"].failure_mode, "off_topic")
        self.assertEqual(self.by_id["refusal"].failure_mode, "refusal")
        for i in ("partial", "offtopic", "refusal"):
            self.assertEqual(self.by_id[i].verdict, "fail")

    def test_low_confidence_goes_to_review(self):
        g = self.by_id["lowconf"]
        self.assertEqual(g.verdict, "review")

    def test_single_batch_call(self):
        self.assertEqual(self.judge.router.calls, 1)

    def test_grades_are_json_serializable(self):
        import json
        from dataclasses import asdict
        json.dumps([asdict(g) for g in self.grades])


class TestCascade(unittest.TestCase):
    def test_review_items_escalate(self):
        grades = grade_with_cascade(LayaJudge(FakeRouter()),
                                    FakeLLMJudge(), ITEMS)
        by_id = {g.item_id: g for g in grades}
        self.assertEqual(by_id["lowconf"].judge, "llm:fake")
        self.assertEqual(by_id["lowconf"].verdict, "pass")
        self.assertIn("first_pass", by_id["lowconf"].detail)
        # Unflagged items are not re-graded
        self.assertEqual(by_id["good"].judge, "laya")


if __name__ == "__main__":
    unittest.main()
