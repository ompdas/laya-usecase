"""PII redaction applied to a ticket *before* any model sees it.

Kept deliberately dependency-free and unit-testable. In production you can
also install the same function as a Laya prediction hook so it runs on every
predict call, including ones outside this pipeline.
"""
from __future__ import annotations

import re

# Order matters: run the card pattern first so the phone pattern cannot
# claim a card number first.
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w.-]+)+\b")
_PHONE = re.compile(r"\b\+?\d[\d -]{8,14}\d\b")

_REPLACEMENTS = (
    (_CARD, "[REDACTED_CARD]"),
    (_EMAIL, "[REDACTED_EMAIL]"),
    (_PHONE, "[REDACTED_PHONE]"),
)


def redact_text(text: str) -> str:
    for pattern, repl in _REPLACEMENTS:
        text = pattern.sub(repl, text)
    return text


def redact_state(state: dict) -> dict:
    """Return a copy of the state with every string value redacted."""
    return {
        key: redact_text(value) if isinstance(value, str) else value
        for key, value in state.items()
    }
