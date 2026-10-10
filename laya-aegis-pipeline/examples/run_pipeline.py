"""Run the full Aegis pipeline over the sample tickets (needs checkpoints).

Run from the repo root:
    python examples/run_pipeline.py
"""
import csv
import json
from pathlib import Path

from laya import Router

from app.pipeline import result_row, run_pipeline

ROOT = Path(__file__).resolve().parent.parent

with open(ROOT / "data" / "sample_tickets.csv", newline="", encoding="utf-8") as f:
    tickets = [{"id": r["id"], "text": r["text"]} for r in csv.DictReader(f)]

router = Router(preload=True)
ctxs = run_pipeline(router, tickets)

for c in ctxs:
    row = result_row(c)
    print(f"{row['id']}: route={row['route']:<16} dept={row['department']:<10} "
          f"redacted={row['redacted']} stages={row['stages']}")

print("\n--- audit trail for one billing ticket (A-1001) ---")
billing = next(c for c in ctxs if c.id == "A-1001")
print(json.dumps(billing.audit, indent=2, ensure_ascii=False))
