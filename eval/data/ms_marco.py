"""MS MARCO passage retrieval subset — real Bing queries with judged passages.

The fair, NLQL-agnostic dimension: both pipelines share the same embedding and
neither can use metadata filters (MS MARCO passages have none). It tests pure
semantic recall — where NLQL is *not* inherently favored — to keep the benchmark
honest.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict

N_QUERIES = 40


def _doc_id(text: str) -> str:
    return "msmarco-" + hashlib.md5(text.encode("utf-8")).hexdigest()[:10]


def load_subset(n: int = N_QUERIES) -> tuple[list[tuple[str, str, dict]], list[dict]]:
    """Stream MS MARCO v2.1 train, return (corpus, queries).

    corpus:  list[(doc_id, text, metadata)]  — deduped passages
    queries: list[{"q", "gold": [doc_id], "answer"}]  — only queries with >=1 judged
             relevant passage (the rest are skipped, as in the standard task).
    """
    from datasets import load_dataset

    ds = load_dataset("microsoft/ms_marco", "v2.1", split="train", streaming=True)
    corpus: "OrderedDict[str, tuple[str, str, dict]]" = OrderedDict()
    queries: list[dict] = []
    for ex in ds:
        if len(queries) >= n:
            break
        pas = ex.get("passages") or {}
        texts = pas.get("passage_text", []) or []
        selected = pas.get("is_selected", []) or []
        gold: list[str] = []
        for t, s in zip(texts, selected):
            t = (t or "").strip()
            if not t:
                continue
            did = _doc_id(t)
            if did not in corpus:
                corpus[did] = (did, t, {"source": "msmarco"})
            if s == 1:
                gold.append(did)
        if not gold:
            continue
        answers = ex.get("answers") or []
        q = (ex.get("query") or "").strip()
        if not q:
            continue
        queries.append(
            {"q": q, "gold": gold, "answer": (answers[0] if answers else "")}
        )
    return list(corpus.values()), queries
