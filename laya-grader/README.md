# LayaJudge — fast, calibrated grading for your eval harness

A drop-in replacement for **LLM-as-judge** on structured grading tasks.
[Laya](https://github.com/NandhaKishorM/laya) answers four grading questions
about every (query, context, response) triple in **one non-autoregressive
forward pass** — ~33 ms, calibrated probabilities, nothing to parse — instead
of one chat completion per item.

> 📰 Companion article draft: [ARTICLE.md](ARTICLE.md)
> 🧭 Part of the **Laya in Production** series:
> [1 · laya-ticket-triage](../laya-ticket-triage) (single-stage triage) →
> [2 · aegis-pipeline](../aegis-pipeline) (decision DAG) →
> **3 · laya-grader (this repo, eval judging)**.

## What it grades

Per eval item, one batched pass answers:

| question | type | meaning |
| --- | --- | --- |
| `goal_achieved` | noul | does the response answer the query? (P) |
| `faithful_to_context` | noul | is every claim supported by the context? (P) |
| `quality` | score | poor / acceptable / excellent |
| `failure_mode` | choice | none · hallucination · incomplete · off_topic · refusal |

Gates turn those into a verdict: `pass` (both P ≥ 0.7 and no failure mode),
`fail`, or `review` (any calibrated confidence < 0.6 — a human or a stronger
judge looks at it).

## The three ways to run it

**1. Pure LayaJudge** — cheapest, fastest, best for CI-scale eval sets:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
python scripts/grade_dataset.py data/sample_eval_set.csv out/grades
```

**2. Head-to-head comparison** — Laya vs your current LLM judge, with
latency, estimated cost, and pass/fail agreement:

```bash
pip install openai && export OPENAI_API_KEY=...
python scripts/compare_judges.py data/sample_eval_set.csv
```

**3. Cascade** — Laya grades everything; only the items it flags `review`
are escalated to the LLM judge (`grade_with_cascade` in
[`grader/judge.py`](grader/judge.py)). You keep the LLM's judgment exactly
where the cheap judge is unsure, at a fraction of the cost.

```python
from laya import Router
from grader.judge import grade_with_cascade
from grader.laya_judge import LayaJudge
from grader.llm_judge import LLMJudge

grades = grade_with_cascade(LayaJudge(Router(preload=True)), LLMJudge(), items)
```

## Why not just use an LLM judge?

| | LLM judge | LayaJudge |
| --- | --- | --- |
| Latency / item | ~0.5–2 s | ~33 ms (all 4 questions, one pass) |
| Output | JSON to parse & schema-check | typed answers + calibrated probs |
| Consistency across runs | prompt- & sampling-sensitive | same questions, same checkpoint |
| Cost at 10k items | real money | minutes of GPU/CPU time |
| Abstention | unreliable | native confidence gate → `review` |
| Multilingual evals | needs a multilingual judge | auto-routed checkpoints |

The honest trade-off: LLM judges are still better for open-ended rubric
*reasoning* ("how well does this essay develop its argument?"). LayaJudge
wins wherever the rubric decomposes into typed checks — which is most
production eval suites.

## Project layout

```
grader/
  judge.py        # Judge protocol + Grade dataclass + cascade helper
  questions.py    # the grading question set + gates
  laya_judge.py   # batched typed-decision grader
  llm_judge.py    # faithful LLM-as-judge stand-in (openai, optional)
scripts/
  grade_dataset.py   # CSV -> grades.csv + grades.jsonl
  compare_judges.py  # latency / cost / agreement table
examples/grade_batch.py
data/sample_eval_set.csv   # 10 items: pass, hallucination, partial,
                           # off-topic, refusal + ES/DE/HI items
tests/test_grader.py       # 7 offline tests with a fake router
```

## Limits

- First run downloads the routed checkpoint (~400 MB); see
  [Laya's README](https://github.com/NandhaKishorM/laya) for device and
  memory tuning.
- Gates (`GOAL_GATE`, `FAITHFUL_GATE`, `REVIEW_CONFIDENCE`) are starting
  points — calibrate on your own gold-labeled evals; Laya's confidence
  shifts with option counts.

## License

MIT — see [LICENSE](LICENSE).
