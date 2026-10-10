"""CLI: run the full pipeline over a CSV, write enriched CSV + audit JSONL.

Usage:
    python scripts/run_pipeline.py data/sample_tickets.csv out/
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from app.pipeline import result_row, run_pipeline


def read_tickets(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return [
            {"id": row.get("id") or str(i), "text": row["text"]}
            for i, row in enumerate(csv.DictReader(f))
        ]


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("usage: run_pipeline.py <input.csv> <output_dir>")
    from laya import Router

    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    tickets = read_tickets(sys.argv[1])
    print(f"Running Aegis pipeline over {len(tickets)} tickets...")
    router = Router(preload=True)
    ctxs = run_pipeline(router, tickets)

    rows = [result_row(c) for c in ctxs]
    with open(out_dir / "routed.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(out_dir / "audit.jsonl", "w", encoding="utf-8") as f:
        for c in ctxs:
            f.write(json.dumps({"id": c.id, "route": c.route, "audit": c.audit},
                               ensure_ascii=False) + "\n")

    from collections import Counter
    print(f"Routes: {dict(Counter(r['route'] for r in rows))}")
    print(f"Wrote {out_dir / 'routed.csv'} and {out_dir / 'audit.jsonl'}")


if __name__ == "__main__":
    main()
