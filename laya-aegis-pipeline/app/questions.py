"""Typed questions for each pipeline stage.

Stage 1 — guard:      is this spam, abuse, or leaking PII?
Stage 2 — triage:     department + urgency + refund/churn signals
Stage 3 — policy:     refund-policy evidence (billing tickets only)
Stage 4 — action:     composed dynamically from the answers of stages 1-3
"""
from __future__ import annotations

GUARD_QUESTIONS = {
    "is_spam": {
        "type": "noul",
        "instructions": "Is this message promotional spam, phishing or a scam?",
    },
    "is_abuse": {
        "type": "noul",
        "instructions": "Does this message contain abuse or harassment aimed at staff?",
    },
    "contains_pii": {
        "type": "noul",
        "instructions": "Does this message contain personal data such as card numbers, "
                        "passwords, phone numbers or email addresses?",
    },
}

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

POLICY_QUESTIONS = {
    "refund_policy_window": {
        "type": "noul",
        "instructions": "Does the ticket indicate the charge is recent enough to "
                        "qualify under a 30-day refund policy?",
    },
    "duplicate_charge_evidence": {
        "type": "noul",
        "instructions": "Is there clear evidence of a duplicate or erroneous charge?",
    },
    "entitlement": {
        "type": "noul",
        "instructions": "Is the sender an active customer asking about their own account?",
    },
}


def build_action_questions(ctx) -> dict:
    """Compose the final action question from the pipeline's findings.

    The *option set itself* is a decision: we only offer `auto_refund` when
    the policy stage produced strong refund evidence, and only offer
    `vip_escalation` when churn risk was flagged. A narrower option set is
    also easier for the model to answer accurately.
    """
    criteria: dict[str, str] = {
        "standard_queue": "routine request that the normal queue can handle",
        "priority_human": "needs a skilled human agent soon",
    }

    refund_evidence = ctx.answers.get("duplicate_charge_evidence", {}).get("noul", 0.0)
    inside_window = ctx.answers.get("refund_policy_window", {}).get("noul", 0.0)
    if refund_evidence >= 0.8 and inside_window >= 0.7:
        criteria["auto_refund"] = (
            "clear duplicate charge inside the refund window; safe to refund automatically"
        )

    churn = ctx.answers.get("churn_risk", {}).get("noul", 0.0)
    if churn >= 0.6:
        criteria["vip_escalation"] = (
            "retention risk; route to the retention team immediately"
        )

    return {
        "action": {
            "type": "choice",
            "instructions": "Given this support ticket and the pipeline's prior findings, "
                            "which single action should be taken?",
            "criteria": criteria,
        }
    }
