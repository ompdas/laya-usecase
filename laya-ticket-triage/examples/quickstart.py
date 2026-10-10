"""Smallest possible Laya triage call (one ticket, one forward pass).

Run from the repo root:
    python examples/quickstart.py
"""
from laya import Router

from app.questions import TRIAGE_QUESTIONS

state = {
    "body": "Hi, we were billed twice for March. Please refund the duplicate "
            "today or we will cancel our plan."
}

router = Router()  # downloads the routed checkpoint on first use
result = router.predict(state, TRIAGE_QUESTIONS)

answers = result["answers"]
print("department        :", answers["department"]["choice"])
print("urgency           :", answers["urgency"]["score"])
print("refund requested  :", answers["refund_requested"]["noul"])
print("churn risk        :", answers["churn_risk"]["noul"])
print("served by         :", result["routing"]["model"])
