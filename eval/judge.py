"""LLM-as-judge: scores a RAG answer against ground-truth points.

Uses a different model from the answer LLM to avoid self-bias.
"""

from __future__ import annotations

import re

from .llm import JUDGE_MODEL, chat


def judge_answer(question: str, points: str, answer: str, model: str | None = None) -> tuple[int, str]:
    prompt = f"""You are an impartial judge evaluating a RAG answer for factual accuracy and coverage.

Question: {question}
Key points a correct answer should cover: {points}
Answer to evaluate: {answer}

Score 0-5:
  5 = fully correct, covers all key points
  4 = mostly correct, minor gap
  3 = partially correct
  2 = mostly wrong or vague
  1 = incorrect
  0 = no answer / unrelated / "insufficient context" with nothing useful

Reply EXACTLY in this format on two lines:
SCORE: <int>
REASON: <one short sentence>"""
    raw = chat(prompt, model=model)
    m = re.search(r"SCORE:\s*(\d+)", raw)
    score = int(m.group(1)) if m else 0
    r = re.search(r"REASON:\s*(.+)", raw)
    reason = (r.group(1).strip() if r else raw.strip().replace("\n", " "))[:200]
    return score, reason


def judge_answer_multi(
    question: str, points: str, answer: str, judges: list[str]
) -> tuple[float, str]:
    """Run several judges and return (average_score, per-judge breakdown).

    Averaging across independent judges cancels single-model noise (e.g. one
    judge occasionally scoring a correct answer 0).
    """
    scores: list[int] = []
    parts: list[str] = []
    for j in judges:
        s, _ = judge_answer(question, points, answer, model=j)
        scores.append(s)
        parts.append(f"{j}={s}")
    avg = sum(scores) / len(scores) if scores else 0.0
    return avg, " ".join(parts)
