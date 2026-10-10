"""Grade the sample eval set and print a verdict table (needs checkpoints).

Run from the repo root:
    python examples/grade_batch.py
"""
import csv
from pathlib import Path

from laya import Router

from grader.laya_judge import LayaJudge

ROOT = Path(__file__).resolve().parent.parent

with open(ROOT / "data" / "sample_eval_set.csv", newline="", encoding="utf-8") as f:
    items = [{"id": r["id"], "query": r["query"],
              "context": r["context"], "response": r["response"]}
             for r in csv.DictReader(f)]

judge = LayaJudge(Router(preload=True))
grades = judge.grade(items)

print(f"{'id':<8} {'verdict':<8} {'goal':<6} {'faith':<6} "
      f"{'quality':<8} {'failure_mode':<14} {'model':<12} {'ms':>6}")
for g in grades:
    print(f"{g.item_id:<8} {g.verdict:<8} {g.goal_achieved:<6} {g.faithful:<6} "
          f"{g.quality:<8} {g.failure_mode:<14} {g.detail.get('model','?'):<12} "
          f"{g.latency_ms:>6.1f}")
