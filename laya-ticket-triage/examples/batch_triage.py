"""Triage a whole CSV in one batched call (shared forward passes).

Run from the repo root:
    python examples/batch_triage.py
"""
import csv
from pathlib import Path

from laya import Router

from app.triage import triage_tickets

ROOT = Path(__file__).resolve().parent.parent

with open(ROOT / "data" / "sample_tickets.csv", newline="", encoding="utf-8") as f:
    tickets = [{"id": r["id"], "text": r["text"]} for r in csv.DictReader(f)]

router = Router(preload=True)  # preload: language flips cost detection only
rows = triage_tickets(router, tickets, batch_size=8, sort_by_length=True)

for row in rows:
    print(
        f"{row['id']}: {row['department']:<10} urgency={row['urgency']:<4} "
        f"churn={row['churn_probability']:<5} model={row['model']:<12} "
        f"review={row['needs_human_review']}"
    )
