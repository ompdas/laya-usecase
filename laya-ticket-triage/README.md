# SmartDesk — multilingual ticket triage with Laya

Route support tickets to the right department, score urgency, and flag refund
intent and churn risk — in **one non-autoregressive forward pass per ticket**,
in 100+ languages, with no prompt engineering and nothing to parse.

Built on [Laya](https://github.com/NandhaKishorM/laya), a typed-decision
("System 1") engine: you ask typed questions (`choice`, `score`, `noul`) over
any state (text, email, ticket, JSON), and a `Router` picks the right
checkpoint (English / multilingual / typed-decisions) per request.

> 📰 The companion article draft lives in [ARTICLE.md](ARTICLE.md).
> 🧭 Part of the **Laya in Production** series: **1 · triage (this repo)** →
> [2 · aegis-pipeline](../aegis-pipeline) (decision DAG) →
> [3 · laya-grader](../laya-grader) (eval judging).

## Why Laya for triage?

| Concern            | LLM-based triage            | Laya                                |
| ------------------ | --------------------------- | ----------------------------------- |
| Latency (1 ticket) | seconds                     | ~33 ms/question, one pass for all   |
| Output             | free text → needs parsing   | typed answers + calibrated probs    |
| Hallucination      | possible                    | no generation; nothing to invent    |
| Multilingual       | needs a multilingual model  | auto-routing across 100+ languages  |
| Adding a question  | re-prompt, re-parse, $$$    | one line of config, same pass       |

## Quickstart

```bash
python -m venv .venv
# Windows: .venv\Scripts\python -m pip install -r requirements.txt
.venv/bin/python -m pip install -r requirements.txt

# First run downloads the routed checkpoint (~400 MB) from Hugging Face.
python examples/quickstart.py
python examples/batch_triage.py          # triage the sample CSV in one batch
```

Offline tests (no model download):

```bash
pip install pytest
pytest tests/
```

### Triage a CSV

```bash
python scripts/triage_csv.py data/sample_tickets.csv out/triaged.csv
```

Output columns: `id, department, urgency, refund_probability,
churn_probability, min_confidence, model, needs_human_review`.

### Run the API

```bash
pip install "laya[serve]"
uvicorn app.main:app --reload            # http://127.0.0.1:8000/docs
```

```bash
curl -s localhost:8000/triage -H 'content-type: application/json' \
  -d '{"text": "We were billed twice for March. Please refund the duplicate today."}'
```

## Project layout

```
app/
  questions.py      # the typed questions (choice / score / noul)
  triage.py         # batching, confidence gating, row shaping
  main.py           # FastAPI service (/triage, /triage/batch)
scripts/
  triage_csv.py     # CSV-in, CSV-out CLI
examples/
  quickstart.py     # smallest possible Laya call
  batch_triage.py   # batched triage of the sample tickets
data/
  sample_tickets.csv  # 8 tickets in 6 languages
tests/
  test_triage.py    # offline tests with a fake router
```

## How it works

1. **Questions, not prompts.** Each ticket is scored against four typed
   questions in a *single* forward pass — adding a question costs a line of
   config, not latency.
2. **Auto-routing.** `Router` detects the script/language in <1 ms and sends
   English to the 421M English checkpoint and everything else to the
   multilingual one — you never manage models per language.
3. **Batched throughput.** `Router.predict_batch` groups compatible requests
   into shared forward passes (measured ~2.6x on 20 tickets) and
   `sort_by_length=True` cuts padding further on mixed-length workloads.
4. **Honest uncertainty.** Answers below `MIN_CONFIDENCE` are escalated to a
   human (`needs_human_review`) instead of being auto-actioned.

## Notes & limits

- First `predict()` downloads a checkpoint (~400 MB); set `HF_HUB_CACHE` to
  move the cache. See [Laya's docs](https://github.com/NandhaKishorM/laya)
  for CPU/GPU/MPS setup and `max_loaded` memory tuning.
- Confidence cutoffs depend on your option counts and workload — calibrate
  `MIN_CONFIDENCE` on your own held-out tickets before auto-routing anything.

## License

MIT — see [LICENSE](LICENSE).
