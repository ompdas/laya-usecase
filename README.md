# Laya in Production — a three-repo series

Three independent, runnable projects that take [Laya](https://github.com/NandhaKishorM/laya)
— the open-source multilingual "System 1" decision engine — from *first typed
decision* to *multi-stage automation pipeline* to *replacing LLM-as-judge in
your eval harness*. Each repo stands alone; together they tell one story:

> **Generation is expensive. Decisions are cheap. Classify the work, and spend
> intelligence only where the decision engine is unsure.**

| # | Repo | Use case | What it demonstrates | Article |
| - | ---- | -------- | -------------------- | ------- |
| 1 | [`laya-ticket-triage`](./laya-ticket-triage) | Multilingual support-ticket triage | Typed questions (`choice` / `score` / `noul`) answered in **one forward pass**; auto-routing across 100+ languages; confidence-based human escalation | [ARTICLE.md](./laya-ticket-triage/ARTICLE.md) |
| 2 | [`aegis-pipeline`](./aegis-pipeline) | Policy-aware ticket automation DAG | **Composed decisions**: stages that short-circuit (quarantine, VIP escalation), mutate state (PII redaction), batch shrinking cohorts, and compute the next question's option set from prior answers — with a full audit trail | [ARTICLE.md](./aegis-pipeline/ARTICLE.md) |
| 3 | [`laya-grader`](./laya-grader) | Eval-harness grading (LLM-judge alternative) | `Judge` protocol with a batched Laya backend and an LLM-judge stand-in; pass/fail/review gates on calibrated confidence; the **cascade** pattern (cheap first pass, LLM only for flagged items) | [ARTICLE.md](./laya-grader/ARTICLE.md) |

## Reading order

- **In a hurry?** Start with repo 1's `examples/quickstart.py` — a working
  Laya call in 20 lines.
- **Architecting something?** Repo 2's `app/pipeline.py` is the pattern
  library: predicates, gates, per-stage batching, audit.
- **Running agent evals?** Repo 3's `compare_judges.py` will grade your own
  dataset head-to-head against your current LLM judge.

## The through-lines

Three ideas recur across all three repos:

1. **One pass, many questions.** Every added question is a line of config,
   not a new API call — the marginal cost of asking *one more thing* is ~0.
2. **Calibrated abstention beats confident guessing.** Every answer carries a
   confidence; all three repos route low-confidence cases to humans (or a
   stronger judge) instead of acting on a coin flip.
3. **Auditability is free.** Because answers are typed values with routing
   metadata, every decision can be logged, replayed, and disputed — no prompt
   transcripts to forensically parse.

## Run everything

```bash
# Each repo is independent. Offline tests need no model download:
cd laya-ticket-triage && python -m unittest discover -s tests   # 7 tests
cd ../aegis-pipeline && python -m unittest discover -s tests    # 14 tests
cd ../laya-grader && python -m unittest discover -s tests       # 7 tests

# Live runs need: python -m pip install laya  (downloads ~400 MB checkpoints once)
```

## Publish all three

```bash
# with GitHub CLI:
for repo in laya-ticket-triage aegis-pipeline laya-grader; do
  (cd "$repo" && gh repo create "$repo" --public --source . --push)
done
```

Or create empty repos on github.com/new and `git remote add origin … && git
push -u origin main` in each directory.

> This file also works as a GitHub **profile README** (`<username>/<username>`)
> or the landing page of a `laya-series` repo if you'd rather present the
> series from a single entry point.

## License

All three repos are MIT. See each repo's `LICENSE`.
