# Your eval harness doesn't need an LLM judge (for the 90%)

> Companion draft for the `laya-grader` repo. Series: triage → pipeline →
> judging. Benchmarks cited are from the
> [Laya repository](https://github.com/NandhaKishorM/laya).

---

Every team running agent evals eventually builds the same thing: a loop that
sends each (input, output) pair to a chat model with a rubric prompt, parses
the JSON, and prays the schema holds. It works. It is also slow (~a second
per item), expensive at CI scale, subtly nondeterministic, and — the part
nobody says out loud — *most of its questions are classification questions
wearing a reasoning costume*.

"Did the agent answer the user's question?" "Is every claim supported by the
retrieved context?" "Which failure mode is this?" These are typed checks,
not essays. And typed checks are exactly what a System-1 decision engine
eats.

## The judge that doesn't generate

[Laya](https://github.com/NandhaKishorM/laya) evaluates typed questions —
`choice`, `score`, `noul` (yes/no as a calibrated probability) — over any
state in a single non-autoregressive forward pass. As a judge, that gives
you four properties an LLM judge can't match:

1. **Latency that disappears from your budget.** ~33 ms for a question on a
   T4; a four-question rubric is one pass, not four calls. A 10,000-item
   eval set grades in minutes of wall clock, and `predict_batch` sharing
   forward passes makes it faster still.
2. **No parsing layer.** Answers arrive as typed values with calibrated
   probabilities. There is no JSON schema to validate, no retry-on-malformed
   loop, no "the judge output `True` as a string" bug at 2 a.m.
3. **Consistency.** Same questions, same checkpoint, every CI run. Your eval
   diffs stop being prompt-drift artifacts.
4. **Calibrated abstention.** LLM judges are notoriously bad at saying "I'm
   not sure." Laya's answers carry per-question confidence, so the harness
   can route unsure items to review instead of silently flipping a coin.

## The rubric

The whole grader config is a dict of questions over the state
`{query, context, response}`:

```python
QA_GRADING_QUESTIONS = {
    "goal_achieved":       {"type": "noul",  "instructions": "...does the response fully answer the query?"},
    "faithful_to_context": {"type": "noul",  "instructions": "...every claim supported by the context?"},
    "quality":             {"type": "score", "criteria": ["poor", "acceptable", "excellent"]},
    "failure_mode":        {"type": "choice","criteria": {"none": ..., "hallucination": ...,
                                                          "incomplete": ..., "off_topic": ..., "refusal": ...}},
}
```

One batched call grades the entire eval set; gates turn the probabilities
into `pass` / `fail` / `review`. Note the failure-mode taxonomy doubles as
your eval analytics: instead of a pass-rate line that moves mysteriously,
you watch hallucinations vs incompletes vs refusals shift between model
versions.

Multilingual eval sets need nothing extra — the `Router` auto-dispatches to
the multilingual checkpoint per item, so German and Hindi items in the same
batch are graded without a separate judge configuration.

## The cascade: where LLM judges still earn their keep

The honest position isn't "replace your LLM judge"; it's **stop paying LLM
prices for items a 33 ms check can grade confidently**. The repo ships
`grade_with_cascade`:

```
every item ──LayaJudge──▶ pass / fail  ── done, $0 marginal cost
                    └──▶ review ──LLM judge──▶ final verdict
```

On a typical eval set the review fraction is small — Laya's benchmarks show
strong zero-shot accuracy on intent/entailment-style tasks — so the cascade
costs a rounding error of a pure-LLM run while keeping human-grade judgment
exactly on the items that need it. The escalation is auditable: review items
carry their first-pass grade in `detail.first_pass`, so you can measure how
often the LLM overturns the cheap judge (if it constantly does, your gates
are miscalibrated).

## Numbers to run yourself

```bash
python scripts/compare_judges.py data/sample_eval_set.csv
```

prints the head-to-head: avg ms/item, estimated judge cost per item, and
pass/fail agreement between LayaJudge and your LLM judge on the same items.
Run it on your *own* eval set before believing anyone's table — including
this article's — because rubrics and item mixes differ.

## When to keep the LLM judge

Open-ended quality ("how persuasive is this generated proposal?"), rubrics
that require multi-step reasoning, and anything where the check itself needs
to read more than a few hundred tokens — keep those on a strong model.
LayaJudge's lane is the long tail of structured checks that make up most
production eval suites, where the winning move is grading *every* item
cheaply instead of a *sample* expensively.

## Try it

```bash
git clone <your-fork-url> && cd laya-grader
python -m venv .venv && .venv/bin/pip install -r requirements.txt
python -m unittest discover -s tests        # 7 offline tests, no download
python scripts/grade_dataset.py data/sample_eval_set.csv out/grades
```

The repo is small on purpose: a question set, two judge backends behind one
protocol, a cascade helper, and tests. Adapting the rubric to your eval is
editing one dict.
