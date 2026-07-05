"""Shared OpenAI-compatible client + config for the eval harness.

All endpoints go through https://ai.su.ki/v1 (OpenAI-compatible).
Credentials come from the environment — never hard-coded.

    export NLQL_EVAL_API_KEY=sk-...
    export NLQL_EVAL_BASE_URL=https://ai.su.ki/v1   # default
"""

from __future__ import annotations

import os

BASE_URL = os.environ.get("NLQL_EVAL_BASE_URL", "https://ai.su.ki/v1")
API_KEY = os.environ.get("NLQL_EVAL_API_KEY")
if not API_KEY:
    raise SystemExit("set NLQL_EVAL_API_KEY before running the eval")

EMBED_MODEL = os.environ.get("NLQL_EVAL_EMBED", "text-embedding-3-small")
CHAT_MODEL = os.environ.get("NLQL_EVAL_CHAT", "doubao-seed-2.0-mini")
JUDGE_MODEL = os.environ.get("NLQL_EVAL_JUDGE", "gpt-5.4")

_client = None


def _client_():
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    return _client


def chat(prompt: str, model: str | None = None, temperature: float = 0.0) -> str:
    """Single-turn chat completion."""
    r = _client_().chat.completions.create(
        model=model or CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return r.choices[0].message.content or ""


def make_nlql_embedder():
    """An OpenAIEmbedder pointed at the same endpoint (shared with LangChain)."""
    from nlql.embed import OpenAIEmbedder

    return OpenAIEmbedder(base_url=BASE_URL, api_key=API_KEY, model=EMBED_MODEL)
