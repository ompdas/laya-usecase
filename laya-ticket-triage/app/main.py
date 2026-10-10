"""FastAPI service exposing ticket triage over HTTP.

Run:
    pip install "laya[serve]"
    uvicorn app.main:app --reload

Then:
    curl -s localhost:8000/triage -H 'content-type: application/json' \
      -d '{"text": "We were billed twice for March. Please refund the duplicate today."}'
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .questions import TRIAGE_QUESTIONS
from .triage import build_requests

app = FastAPI(title="SmartDesk triage", version="0.1.0")

_router = None


def get_router():
    """Lazy singleton so uvicorn workers don't each load the checkpoints twice."""
    global _router
    if _router is None:
        from laya import Router

        _router = Router(
            preload=True,
            device=os.getenv("LAYA_DEVICE") or None,  # "cuda" | "mps" | "cpu"
        )
    return _router


class TriageRequest(BaseModel):
    text: str = Field(..., examples=["We were billed twice, please refund."])


class BatchRequest(BaseModel):
    tickets: list[TriageRequest]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/triage")
def triage_one(body: TriageRequest):
    router = get_router()
    requests = build_requests([{"id": 0, "text": body.text}])
    result = router.predict_batch(requests)[0]
    return {
        "answers": result["answers"],
        "routing": result["routing"],
    }


@app.post("/triage/batch")
def triage_batch(body: BatchRequest):
    router = get_router()
    tickets = [{"id": i, "text": t.text} for i, t in enumerate(body.tickets)]
    requests = build_requests(tickets)
    results = router.predict_batch(requests, batch_size=16, sort_by_length=True)
    return {
        "results": [
            {"answers": r["answers"], "routing": r["routing"]} for r in results
        ]
    }
