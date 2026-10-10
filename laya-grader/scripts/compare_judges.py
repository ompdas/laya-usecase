"""Compare LayaJudge against an LLM judge on the same eval set.

Laya side always runs (needs checkpoints). The LLM side runs only if
OPENAI_API_KEY is set — otherwise the script says so and reports Laya only.

Usage:
    python scripts/compare_judges.py data/sample_eval_set.csv
"""
from __future__ import annotations

import os
import sys

from grade_dataset import read_items


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_eval_set.csv"
    items = read_items(path)

    from laya import Router
    from grader.laya_judge import LayaJudge

    laya = LayaJudge(Router(preload=True))
    laya_grades = laya.grade(items)

    print("\n=== LayaJudge ===")
    print(f"  items: {len(laya_grades)}")
    print(f"  avg latency: {sum(g.latency_ms for g in laya_grades)/len(laya_grades):.1f} ms/item")
    print(f"  verdicts: { {v: sum(1 for g in laya_grades if g.verdict==v) for v in ('pass','fail','review')} }")

    if not os.getenv("OPENAI_API_KEY"):
        print("\nOPENAI_API_KEY not set — skipping LLM judge comparison.")
        print("Set it (and pip install openai) to see the cost/latency table.")
        return

    from grader.llm_judge import LLMJudge

    llm = LLMJudge()
    llm_grades = llm.grade(items)
    print("\n=== LLM judge ===")
    print(f"  model: {llm.model}")
    print(f"  avg latency: {sum(g.latency_ms for g in llm_grades)/len(llm_grades):.1f} ms/item")
    print(f"  est. cost: ${sum(g.detail.get('est_cost_usd', 0) for g in llm_grades):.6f}")

    agree = sum(1 for a, b in zip(laya_grades, llm_grades)
                if (a.verdict == "pass") == (b.verdict == "pass"))
    print(f"\nPass/fail agreement: {agree}/{len(items)} "
          f"({agree/len(items):.0%})")


if __name__ == "__main__":
    main()
