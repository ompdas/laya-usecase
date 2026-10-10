"""Typed questions for support-ticket triage.

Laya answers every question in the *same* forward pass, so adding a
question costs ~nothing in latency — only a line of configuration.
"""

TRIAGE_QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
            "billing": "invoices, payments, refunds, duplicate charges",
            "technical": "bugs, outages, errors, login problems",
            "sales": "pricing, plans, contracts, upgrades",
            "other": "everything else",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"],
    },
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the user explicitly request a refund?",
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel, churn or leave?",
    },
}

# Confidence below this sends the ticket to a human instead of an auto-action.
MIN_CONFIDENCE = 0.60
