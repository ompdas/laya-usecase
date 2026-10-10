"""Grading question sets.

Each item's state is {"query", "context", "response"}; the instructions name
the fields so the model knows where to look. All four questions are answered
in ONE forward pass per item.
"""

QA_GRADING_QUESTIONS = {
    "goal_achieved": {
        "type": "noul",
        "instructions": "Given the 'query' field (what the user asked) and the "
                        "'response' field (what the system answered): does the "
                        "response fully answer the user's query?",
    },
    "faithful_to_context": {
        "type": "noul",
        "instructions": "Is every factual claim in the 'response' field supported "
                        "by the 'context' field? Answer no if the response states "
                        "facts that are not in the context.",
    },
    "quality": {
        "type": "score",
        "instructions": "How good is the 'response' as an answer to the 'query'?",
        "criteria": ["poor", "acceptable", "excellent"],
    },
    "failure_mode": {
        "type": "choice",
        "instructions": "If the response fails, what is its primary failure mode? "
                        "Pick 'none' only if it succeeds.",
        "criteria": {
            "none": "the response answers the query correctly",
            "hallucination": "states facts that are not in the context",
            "incomplete": "answers only part of the query",
            "off_topic": "does not address the query",
            "refusal": "refuses or deflects when it should have answered",
        },
    },
}

# Gates: tune on your own held-out grades. `review` wins over `pass`.
GOAL_GATE = 0.70
FAITHFUL_GATE = 0.70
REVIEW_CONFIDENCE = 0.60
