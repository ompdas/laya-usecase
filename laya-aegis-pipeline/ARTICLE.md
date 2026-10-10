# Beyond triage: composing a multi-stage decision pipeline with Laya

> Companion draft for the `aegis-pipeline` repo. Follows the simpler
> `laya-ticket-triage` demo; assumes the reader knows what a typed decision
> (`choice` / `score` / `noul`) is. Benchmarks cited are from the
> [Laya repository](https://github.com/NandhaKishorM/laya).

---

A single Laya call answers several questions about one input in one forward
pass. That's already useful. But the more interesting architecture question
is: what happens when **the answer to one decision changes which decisions
you ask next**? Real support operations aren't one classifier — they're a
pipeline with gates, escalations, policy checks, and a final action that
depends on everything that came before.

This article builds **Aegis**: a four-stage decision DAG where every stage is
a Laya batch, and shows three patterns that fall out naturally — cohort
batching, computed option sets, and decision audit trails.

## The DAG

```
ticket → [guard] → [triage] → [policy]? → [action] → route
           │spam      │churn       │billing
           ▼          ▼            ▼
      quarantine  vip_escalation  (skipped otherwise)
```

- **guard** — three `noul` checks: spam, abuse, PII. Spam or abuse ≥ 0.8
  ends the ticket in `quarantine`. PII ≥ 0.7 triggers *state mutation*: the
  ticket is redacted (card numbers, emails, phones) before any later model
  sees it.
- **triage** — department (`choice`), urgency (`score`), refund intent and
  churn risk (`noul`). Churn ≥ 0.85 short-circuits to `vip_escalation`.
- **policy** — runs **only** for billing tickets (a predicate, not an
  `if` in business code): refund-window, duplicate-charge evidence,
  entitlement.
- **action** — a final `choice` whose **criteria are composed from the
  answers of the previous stages**.

## Pattern 1: stages are batches, and the cohort shrinks

The engine doesn't loop over tickets. Per stage, it collects the tickets
that are still alive *and* match the stage predicate, and scores them in one
`Router.predict_batch` call — one forward pass per stage per ticket, shared
across the cohort. On the sample data (10 tickets), the batch sizes are:

| stage  | tickets batched | why it shrank                          |
| ------ | --------------- | -------------------------------------- |
| guard  | 8 tested        | —                                      |
| triage | 6               | spam + abuse quarantined               |
| policy | 3               | only billing, churn ticket escalated   |
| action | 5               | quarantined and escalated tickets gone |

This matters economically: with the Laya team's measured 7.2 ms/question
batched on a T4, a four-question stage costs ~30 ms of GPU time per ticket,
and dead tickets never pay for downstream stages at all.

## Pattern 2: the option set is computed, not static

The action stage is where the pipeline stops being a config file:

```python
def build_action_questions(ctx):
    criteria = {
        "standard_queue": "routine request the normal queue can handle",
        "priority_human": "needs a skilled human agent soon",
    }
    if refund_evidence >= 0.8 and inside_window >= 0.7:
        criteria["auto_refund"] = "clear duplicate charge inside the window"
    if churn >= 0.6:
        criteria["vip_escalation"] = "retention risk; retention team now"
    return {"action": {"type": "choice", "criteria": criteria, ...}}
```

Two payoffs. Safety: `auto_refund` is *unofferable* without policy evidence —
the model can't pick an action the pipeline hasn't justified. Accuracy: a
`choice` question with 4 sharp options is easier to answer correctly than
one with 12, and Laya's confidence behaves better on narrower sets (the docs
are explicit that confidence shifts with option counts — calibrate on your
data).

## Pattern 3: every decision is auditable

Each stage appends a record:

```json
{"stage": "policy", "model": "typed-decisions",
 "reason": "explicit task routing", "latency_ms": 31.4,
 "answers": {"duplicate_charge_evidence": {"type": "noul", "value": 0.93,
                                           "confidence": 0.9}, ...}}
```

When someone asks "why did the machine refund this ticket?", the answer is a
JSONL file, not a prompt transcript. Note the `model` field: because the
`Router` routes per request, a single pipeline run may be served by different
checkpoints for different tickets (English vs multilingual), and the audit
trail shows exactly which one decided what, and why.

## The honesty section

Three things this pipeline deliberately does *not* do:

1. **No auto-execution.** `auto_refund` is a *recommendation*; wire it to
   your refund service only behind your own idempotency and limits, and only
   after calibrating the gates on held-out tickets.
2. **No long-document reasoning.** A fast System-1 router is the wrong tool
   for a three-page contract dispute — gates should escalate those.
3. **No silent low-confidence automation.** Every answer carries a calibrated
   confidence; the repo's row output exposes `min_confidence` per ticket so
   unsure tickets can be flagged for review in your dashboard.

## Try it

```bash
git clone <your-fork-url> && cd aegis-pipeline
python -m venv .venv && .venv/bin/pip install -r requirements.txt
python -m unittest discover -s tests     # 14 offline tests, no download
python examples/run_pipeline.py          # live run, ~400 MB first time
```

If you build something with composed option sets or per-stage batching, I'd
genuinely like to see the DAG you end up with.
