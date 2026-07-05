"""Shared OpenAI-compatible client + config for the eval harness.

All endpoints go through https://ai.su.ki/v1 (OpenAI-compatible).
Credentials come from the environment — never hard-coded.

    export NLQL_EVAL_API_KEY=sk-...
    export NLQL_EVAL_BASE_URL=https://ai.su.ki/v1   # default
"""

from __future__ import annotations

import hashlib
import os

BASE_URL = os.environ.get("NLQL_EVAL_BASE_URL", "https://ai.su.ki/v1")
API_KEY = os.environ.get("NLQL_EVAL_API_KEY")
if not API_KEY:
    raise SystemExit("set NLQL_EVAL_API_KEY before running the eval")

EMBED_MODEL = os.environ.get("NLQL_EVAL_EMBED", "text-embedding-3-small")
CHAT_MODEL = os.environ.get("NLQL_EVAL_CHAT", "qwen3.7-plus")
JUDGE_MODELS = [
    m.strip()
    for m in os.environ.get("NLQL_EVAL_JUDGES", "qwen3.7-max,deepseek-v4-flash,minimax-m3").split(",")
    if m.strip()
]
JUDGE_MODEL = JUDGE_MODELS[0] if JUDGE_MODELS else "qwen3.7-max"  # back-compat single

_client = None


def _client_():
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    return _client


_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")


def chat(prompt: str, model: str | None = None, temperature: float = 0.0) -> str:
    """Single-turn chat completion.

    Deterministic calls (temperature=0) are cached to disk under eval/.cache/,
    so re-runs skip unchanged prompts and don't burn API quota.
    """
    model = model or CHAT_MODEL
    cache_path: str | None = None
    if temperature == 0.0:
        key = hashlib.sha1(f"{model}\x00{prompt}".encode("utf-8")).hexdigest()
        cache_path = os.path.join(_CACHE_DIR, key + ".txt")
        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                return f.read()
    r = _client_().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    resp = r.choices[0].message.content or ""
    if cache_path is not None:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(resp)
    return resp


def make_nlql_embedder():
    """An OpenAIEmbedder pointed at the same endpoint (shared with LangChain)."""
    from nlql.embed import OpenAIEmbedder

    return OpenAIEmbedder(base_url=BASE_URL, api_key=API_KEY, model=EMBED_MODEL, timeout=120.0)
