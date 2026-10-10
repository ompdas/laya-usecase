"""CLI: triage a CSV of tickets and write an enriched CSV back.

Usage:
    python scripts/triage_csv.py data/sample_tickets.csv /tmp/triaged.csv
"""
from __future__ import annotations

import csv
import sys

from app.questions import MIN_CONFIDENCE
from app.triage import triage_tickets


def read_tickets(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return [
            {"id": row.get("id") or str(i), "text": row["text"]}
            for i, row in enumerate(csv.DictReader(f))
        ]


def write_rows(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("usage: triage_csv.py <input.csv> <output.csv>")
    from laya import Router

    tickets = read_tickets(sys.argv[1])
    print(f"Triageing {len(tickets)} tickets (min confidence {MIN_CONFIDENCE})...")
    router = Router(preload=True)
    rows = triage_tickets(router, tickets, batch_size=16)
    write_rows(sys.argv[2], rows)
    escalated = sum(r["needs_human_review"] for r in rows)
    print(f"Wrote {len(rows)} rows to {sys.argv[2]} ({escalated} escalated to humans).")


if __name__ == "__main__":
    main()
