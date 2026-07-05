"""BEIR/scifact subset — public scientific IR benchmark.

Complements MS MARCO with a different domain (scientific fact-checking abstracts).
Same fair shape: both pipelines share the embedding, passages carry no metadata,
so NLQL's filter advantage does not apply — this is a second pure-recall data point.
"""

from __future__ import annotations

import random

N_QUERIES = 100
N_DISTRACTORS = 1000
_SEED = 7


def load_subset(n_queries: int = N_QUERIES, n_distractors: int = N_DISTRACTORS):
    """Return (corpus, queries) sampled from BEIR/scifact.

    corpus:  list[(doc_id, text, metadata)]  — gold passages for sampled queries
             plus N_DISTRACTORS random passages
    queries: list[{"q", "gold": [doc_id]}]    — test queries with judged relevant docs
    """
    from datasets import load_dataset

    corpus_ds = load_dataset("BeIR/scifact", "corpus", split="corpus")
    queries_ds = load_dataset("BeIR/scifact", "queries", split="queries")
    qrels = load_dataset("BeIR/scifact-qrels", split="test")

    gold: dict[int, list[int]] = {}
    for r in qrels:
        gold.setdefault(r["query-id"], []).append(r["corpus-id"])

    corpus_by_id = {int(c["_id"]): c for c in corpus_ds}
    rng = random.Random(_SEED)
    cand: list[tuple[int, dict]] = []
    for q in queries_ds:
        try:
            qid = int(q["_id"])
        except (ValueError, KeyError, TypeError):
            continue
        if qid in gold and all(g in corpus_by_id for g in gold[qid]):
            cand.append((qid, q))
    rng.shuffle(cand)
    cand = cand[:n_queries]

    needed: set[str] = set()
    queries: list[dict] = []
    for i, q in cand:
        needed.update(gold[i])
        queries.append(
            {
                "q": (q.get("text") or q.get("title") or "").strip(),
                "gold": [f"scifact-{g}" for g in gold[i]],
            }
        )

    all_ids = list(corpus_by_id)
    rng.shuffle(all_ids)
    for cid in all_ids:
        if len(needed) >= n_queries + n_distractors:
            break
        needed.add(cid)

    docs: list[tuple[str, str, dict]] = []
    for cid in needed:
        c = corpus_by_id[cid]
        docs.append(
            (
                f"scifact-{cid}",
                ((c.get("title") or "") + " " + (c.get("text") or "")).strip(),
                {"source": "scifact"},
            )
        )
    return docs, queries
