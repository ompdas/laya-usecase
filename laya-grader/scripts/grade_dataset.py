"""CLI: grade an eval CSV with LayaJudge, write grades CSV + JSONL.

Usage:
    python scripts/grade_dataset.py data/sample_eval_set.csv out/grades
"""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

from grader.judge import grade_with_cascade
from grader.laya_judge import LayaJudge


def read_items(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return [
            {"id": r["id"], "query": r["query"],
             "context": r["context"], "response": r["response"]}
            for r in csv.DictReader(f)
        ]


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("usage: grade_dataset.py <input.csv> <output_prefix>")
    from laya import Router

    items = read_items(sys.argv[1])
    print(f"Grading {len(items)} items with LayaJudge...")
    judge = LayaJudge(Router(preload=True))
    grades = judge.grade(items)

    prefix = Path(sys.argv[2])
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{prefix}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(grades[0]).keys()))
        writer.writeheader()
        for g in grades:
            row = asdict(g)
            row["detail"] = json.dumps(row["detail"], ensure_ascii=False)
            writer.writerow(row)
    with open(f"{prefix}.jsonl", "w", encoding="utf-8") as f:
        for g in grades:
            f.write(json.dumps(asdict(g), ensure_ascii=False) + "\n")

    from collections import Counter
    print(f"Verdicts: {dict(Counter(g.verdict for g in grades))}")
    print(f"Avg latency: {sum(g.latency_ms for g in grades)/len(grades):.1f} ms/item")
    print(f"Wrote {prefix}.csv and {prefix}.jsonl")


if __name__ == "__main__":
    main()
