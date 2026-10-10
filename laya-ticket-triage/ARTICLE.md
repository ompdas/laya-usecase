# Every ticket, every language, one forward pass: building a multilingual support-triage pipeline with Laya

> Companion article draft for the `laya-ticket-triage` repo. Tone/length targets
> a technical blog (Dev.to / Medium / company engineering blog). Trim the
> "Notes" as needed. Benchmarks cited are from the
> [Laya repository](https://github.com/NandhaKishorM/laya) and marked as such.

---

Support triage looks easy until you price it. A mid-size SaaS might get a few
hundred tickets a day, in a handful of languages, and every minute a billing
complaint sits in the wrong queue is a minute of churn risk compounding. The
obvious modern answer is "just use an LLM": send each ticket to GPT/Claude
with a clever prompt, parse the JSON, hope the model doesn't improvise a new
department name on a bad day. It works — at seconds of latency per ticket,
non-trivial cost at volume, and a parsing layer you now own forever.

This article is about a different tool for this job: **Laya**, an open-source
"System 1" decision engine. Instead of generating text, Laya evaluates
*typed questions* over your input in a single non-autoregressive forward pass —
33 ms per question on a T4, 7.2 ms/question batched, measured by the Laya
team — and returns structured answers with calibrated probabilities. Nothing
is generated, so nothing needs to be parsed and nothing can be hallucinated.

We'll build **SmartDesk**: a small pipeline that reads a CSV of multilingual
support tickets, answers four questions about each one in one pass, and writes
back an enriched CSV with an auto-routing decision — plus a FastAPI service
version. The whole core is about 100 lines of Python.

## The questions are the program

Laya doesn't take prompts. It takes a dict of *typed questions* over a
*state* (any string, email, ticket, or JSON document). Three question types
cover most triage needs:

- `choice` — pick one label from a criteria dict ("which department?")
- `score` — an ordinal level ("how urgent?")
- `noul` — a yes/no statement check, returned as P(true) ("does the user
  threaten to churn?")

```python
TRIAGE_QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
            "billing": "invoices, payments, refunds, duplicate charges",
            "technical": "bugs, outages, errors, login problems",
            "sales": "pricing, plans, contracts, upgrades",
            "other": "everything else",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"],
    },
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the user explicitly request a refund?",
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel, churn or leave?",
    },
}
```

The detail worth pausing on: **all four questions are answered in the same
forward pass.** With an LLM you'd pay per question (or juggle a fragile
"answer everything as JSON" prompt); with Laya, "should we also detect refund
intent?" is a one-line config change with ~zero latency cost.

## One call, any language

The `Router` is the entry point. It detects the script and language of each
state in under a millisecond and dispatches to the right checkpoint: a 421M
English ModernBERT for identified English, a 322M multilingual mmBERT for
everything else — 100+ languages.

```python
from laya import Router
from app.questions import TRIAGE_QUESTIONS

router = Router(preload=True)   # checkpoints resident; flips cost detection only

res_en = router.predict(
    {"body": "We were billed twice for March. Please refund the duplicate today "
             "or we will cancel our plan."},
    TRIAGE_QUESTIONS,
)
res_hi = router.predict(
    {"body": "मुझसे मार्च में दो बार शुल्क लिया गया, कृपया डुप्लिकेट राशि वापस करें।"},
    TRIAGE_QUESTIONS,
)

res_en["answers"]["department"]["choice"]   # 'billing'
res_hi["answers"]["department"]["choice"]   # 'billing'
res_hi["routing"]["model"]                  # 'multilingual'
```

Note what you did *not* do: no per-language model selection, no language
parameter, no "if lang == 'hi'" branch. The routing metadata even tells you
*why* a checkpoint was picked ("non-Latin script (devanagari, 100% of
letters); the English checkpoint cannot read it").

This routing isn't cosmetic. On the Laya team's 17k-question benchmark, the
English checkpoint collapses outside English (0.306 accuracy on non-English
MASSIVE-intent vs 0.451 for the multilingual one, and 0.000 raw on Khmer at
0.952 confidence — confident *and* wrong, which is the worst failure mode for
automation). The Router gets the multilingual model's 48/51 usable languages
*and* the English model's best-in-class English accuracy, by routing rather
than averaging.

## Batching: throughput is a feature

Per-ticket `predict()` leaves the batch dimension idle. `Router.predict_batch`
routes the whole workload first, groups compatible requests, and shares
forward passes — measured **2.6x faster on 20 tickets** than looping
`predict()`, per the Laya benchmarks. On mixed-length workloads,
`sort_by_length=True` groups similar-sized states so each pass pads less
(measured 1.42x on a real 128-review workload, decisions unchanged).

```python
from app.triage import triage_tickets

rows = triage_tickets(router, tickets, batch_size=8, sort_by_length=True)
# -> [{'id': 'T-1001', 'department': 'billing', 'urgency': 2.0,
#      'refund_probability': 0.92, 'churn_probability': 0.89,
#      'min_confidence': 0.88, 'model': 'english',
#      'needs_human_review': False}, ...]
```

At 7.2 ms/question batched, a four-question ticket is ~30 ms of GPU time.
A 1,000-ticket overnight backlog is seconds, not an hour of API calls.

## Confidence is a product feature, not a metric

Every answer carries a calibrated confidence, and `noul` answers come back as
probabilities, so automation thresholds become product decisions:

```python
MIN_CONFIDENCE = 0.60   # below this, a human sees the ticket

def needs_human_review(answers):
    return any(answer_confidence(a) < MIN_CONFIDENCE for a in answers.values())
```

One honest caveat the Laya docs are admirably blunt about: confidence values
shift with the number of options in a question, so you must fit and validate
your cutoff on *your* held-out tickets. A triage system that silently
auto-routes low-confidence answers is worse than no triage system; the
abstention path is the feature.

## Serving it

The repo wraps the same logic in a minimal FastAPI service:

```bash
pip install "laya[serve]"
uvicorn app.main:app --reload
```

```bash
curl -s localhost:8000/triage -H 'content-type: application/json' \
  -d '{"text": "We were billed twice for March. Please refund the duplicate today."}'
```

In production you'd preload the Router once per worker (the repo uses a lazy
singleton), pin `LAYA_DEVICE=cuda` where available, and keep
`max_loaded` in mind: the lazy default keeps two checkpoints resident, which
is exactly what auto-routing needs.

## Laya, or an LLM?

Triage is a classification-shaped problem wearing a language-shaped costume,
and that profile is Laya's sweet spot. A quick decision table:

| Concern | LLM-based triage | Laya |
| --- | --- | --- |
| Latency per ticket | seconds | ~33 ms/question, one pass for all four |
| Output contract | free text → JSON parsing | typed answers + calibrated probabilities |
| Hallucination surface | prompt can be ignored | no generation at all |
| Multilingual | one big multilingual model, one price | auto-routed specialist checkpoints |
| Adding a question | new prompt, new parser path | one line of config |
| Long, reasoning-heavy tickets | genuinely better | out of scope — escalate these |

The last row is the honest one: Laya is a fast, cheap *router* for the 90% of
tickets that are routine. The long, weird, emotionally complex ones should
still go to a human (or an LLM) — and now your `needs_human_review` flag tells
you which ones.

## Try it

```bash
git clone <your-fork-url> && cd laya-ticket-triage
python -m venv .venv && .venv/bin/pip install -r requirements.txt
python examples/batch_triage.py     # first run downloads ~400 MB of checkpoints
pytest tests/                       # offline logic tests, no download
```

Full code, sample data (8 tickets, 6 languages), and the CSV CLI are in the
repo. If you adapt this to your own queues, calibrate `MIN_CONFIDENCE` on
held-out data before letting it auto-route — and tell me what you built.
