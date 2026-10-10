"""LLMJudge: the classic LLM-as-judge, kept as a comparison and escalation backend.

This is deliberately a faithful stand-in for what teams run today: one chat
completion per item, a rubric prompt, JSON output parsed back into grades.
Costs are *estimated* from token counts and configurable prices — enough to
make the comparison concrete without needing billing access.

Requires: pip install openai  and  OPENAI_API_KEY in the environment.
"""
from __future__ import annotations

import json
import os
import time

from .judge import Grade
from .questions import GOAL_GATE, FAITHFUL_GATE

# Rough pricing defaults (USD per 1M tokens) — override via env.
INPUT_PRICE_PER_MTOK = float(os.getenv("JUDGE_INPUT_PRICE", "0.15"))
OUTPUT_PRICE_PER_MTOK = float(os.getenv("JUDGE_OUTPUT_PRICE", "0.60"))

_RUBRIC = """You are grading an AI system's answer. Respond with JSON only:
{"goal_achieved": true|false, "faithful": true|false,
 "quality": 0|1|2, "failure_mode": "none"|"hallucination"|"incomplete"|"off_topic"|"refusal"}

query: {query}
context: {context}
response: {response}"""


class LLMJudge:
    name = "llm"

    def __init__(self, model: str = "gpt-4o-mini"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("LLMJudge needs the openai package: pip install openai")
        self.client = OpenAI()
        self.model = model

    def _prompt(self, item: dict) -> str:
        return _RUBRIC.format(query=item["query"], context=item["context"],
                              response=item["response"])

    @staticmethod
    def _est_cost(text: str) -> float:
        return (len(text) / 4) / 1_000_000 * INPUT_PRICE_PER_MTOK

    def grade(self, items: list[dict]) -> list[Grade]:
        grades = []
        for item in items:
            prompt = self._prompt(item)
            t0 = time.perf_counter()
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            raw = resp.choices[0].message.content
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = {"goal_achieved": False, "faithful": False,
                          "quality": 0, "failure_mode": "off_topic"}
            goal = 1.0 if parsed.get("goal_achieved") else 0.0
            faithful = 1.0 if parsed.get("faithful") else 0.0
            passed = (goal >= GOAL_GATE and faithful >= FAITHFUL_GATE
                      and parsed.get("failure_mode") == "none")
            est_cost = self._est_cost(prompt) + self._est_cost(raw or "")
            grades.append(Grade(
                item_id=item["id"],
                judge=f"{self.name}:{self.model}",
                verdict="pass" if passed else "fail",
                goal_achieved=goal,
                faithful=faithful,
                quality=float(parsed.get("quality", 0)),
                failure_mode=parsed.get("failure_mode", "off_topic"),
                min_confidence=None,          # raw LLM judges don't give one
                latency_ms=round(latency_ms, 1),
                detail={"est_cost_usd": round(est_cost, 8),
                        "usage": resp.usage.model_dump() if resp.usage else None},
            ))
        return grades
