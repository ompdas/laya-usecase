"""FastAPI service exposing the full Aegis pipeline.

Run:
    pip install "laya[serve]"
    uvicorn app.main:app --reload

    curl -s localhost:8000/pipeline -H 'content-type: application/json' -d '{
      "tickets": [{"id": "A-1", "text": "We were billed twice, refund or we cancel."}]
    }'
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .pipeline import result_row, run_pipeline

app = FastAPI(title="Aegis pipeline", version="0.1.0")

_router = None


def get_router():
    global _router
    if _router is None:
        from laya import Router

        _router = Router(
            preload=True,
            device=os.getenv("LAYA_DEVICE") or None,
        )
    return _router


class TicketIn(BaseModel):
    id: str = Field(..., examples=["A-1001"])
    text: str = Field(..., examples=["We were billed twice, please refund."])


class PipelineRequest(BaseModel):
    tickets: list[TicketIn]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/pipeline")
def pipeline_endpoint(body: PipelineRequest):
    router = get_router()
    tickets = [{"id": t.id, "text": t.text} for t in body.tickets]
    ctxs = run_pipeline(router, tickets)
    return {
        "results": [
            {"row": result_row(c), "audit": c.audit, "redacted_state": c.state}
            for c in ctxs
        ]
    }
