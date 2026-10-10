# Aegis — a multi-stage, policy-aware automation pipeline on Laya

Not just triage: a full **decision DAG** for incoming support tickets. Every
ticket flows through guard → triage → policy → action, where each stage is a
typed-decision batch scored by [Laya](https://github.com/NandhaKishorM/laya)
in a single non-autoregressive forward pass — and each stage can
short-circuit the ticket, redact its PII, or reshape the questions the next
stage asks.

> 📰 Companion article draft: [ARTICLE.md](ARTICLE.md)
> 🧭 Part of the **Laya in Production** series:
> [1 · laya-ticket-triage](../laya-ticket-triage) (single-stage triage) →
> **2 · aegis-pipeline (this repo, decision DAG)** →
> [3 · laya-grader](../laya-grader) (eval judging).

## The pipeline

```
            ticket
              │
           [guard] ── is_spam / is_abuse ≥ 0.8 ──▶ quarantine
              │ pass (redact state if PII ≥ 0.7)
           [triage] ── churn_risk ≥ 0.85 ──▶ vip_escalation
              │  dept · urgency · refund · churn
     billing? ──yes──▶ [policy] ── refund window · duplicate evidence · entitlement ─┐
        no ─────────────────────────────────────────────────────────────────────────┤
                                                                                     ▼
                                            [action]  options composed from findings
                                              │
              auto_refund · priority_human · vip_escalation · standard_queue ...
```

Three ideas make this more than a classifier:

1. **Stages are batches, not loops.** Every ticket that reaches a stage is
   scored against that stage's questions in one shared `predict_batch`
   forward pass. The cohort shrinks as gates fire — in the sample data, 8
   tickets enter the guard, 6 reach triage, 3 reach policy, 5 reach action.
2. **The option set is computed, not static.** The action stage's `choice`
   criteria are built from earlier answers: `auto_refund` is only offered
   when policy evidence is strong; `vip_escalation` only when churn was
   flagged. Fewer, sharper options are both safer and more accurate.
3. **Every decision is auditable.** Each stage records the model that served
   it, the routing reason (e.g. why a ticket went to the multilingual
   checkpoint), per-question values with calibrated confidence, and latency —
   written to `audit.jsonl`.

## Quickstart

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt   # Windows: .venv\Scripts\python

# Offline tests first — no model download needed (14 tests, all passing):
python -m unittest discover -s tests

# Live run — downloads checkpoints (~400 MB) on first use:
python examples/run_pipeline.py
python scripts/run_pipeline.py data/sample_tickets.csv out/
```

### API

```bash
pip install "laya[serve]"
uvicorn app.main:app --reload        # http://127.0.0.1:8000/docs
```

```bash
curl -s localhost:8000/pipeline -H 'content-type: application/json' -d '{
  "tickets": [{"id": "A-1",
               "text": "We were billed twice, refund the duplicate or we cancel."}]
}'
```

The response carries the final route, the flattened row, the full per-stage
audit trail, and the (redacted) state the models actually saw — which is
what you want for debugging a decision you disagree with.

## Project layout

```
app/
  redact.py         # PII redaction applied before any model sees the state
  questions.py      # guard / triage / policy questions + composed action set
  pipeline.py       # DAG engine: predicates, gating, per-stage batching, audit
  main.py           # FastAPI service
scripts/run_pipeline.py   # CSV in -> routed.csv + audit.jsonl
examples/run_pipeline.py  # live demo with audit pretty-print
data/sample_tickets.csv   # 10 tickets: spam, abuse, PII, churn, 5 languages
tests/test_pipeline.py    # 14 offline tests with a fake router
```

## Tuning & limits

- Gate thresholds (`SPAM_GATE`, `CHURN_ESCALATION_GATE`, …) in
  `app/pipeline.py` are illustrative. Calibrate them on your own held-out
  tickets; Laya's confidence values shift with option counts, so fit cutoffs
  at the option counts you actually use.
- First `predict` downloads the routed checkpoint (~400 MB); set
  `HF_HUB_CACHE` to move the cache. See
  [Laya's README](https://github.com/NandhaKishorM/laya) for device
  selection (`LAYA_DEVICE=cuda|mps|cpu`) and `max_loaded` memory tuning.
- Long, reasoning-heavy tickets are out of scope for a fast router — design
  your gates so those escalate to humans (or an LLM) instead.

## License

MIT — see [LICENSE](LICENSE).
