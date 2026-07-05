"""Two-stage retrieval: coarse vector recall + precise reranking.

Run: python examples/reranking.py   (offline; FakeEmbedder + FakeReranker)

The vector stage is a bi-encoder — a short chunk's embedding vs a long query's embedding is
only a rough match. A reranker re-scores each ``(query, passage)`` pair *jointly* and reorders
the over-fetched candidates. Swap in ``nlql.rerank.CrossEncoderReranker()`` for a real cross-encoder.
"""

from __future__ import annotations

import nlql
from nlql.embed import FakeEmbedder
from nlql.rerank import FakeReranker

DOCS = [
    # Contains every query term but is long, so its bi-encoder score is diluted.
    ("agent memory planning tool retrieval vector index query model system", "full"),
    ("banana bread recipe with flour and sugar", "noise"),
    ("agent", "partial"),
]
QUERY = 'SELECT SENTENCE LET rel = SIMILARITY(content, "agent memory planning tool") ORDER BY rel DESC LIMIT 3'


def main() -> None:
    for label, reranker in [("without reranking", None), ("with FakeReranker", FakeReranker())]:
        engine = nlql.Engine(FakeEmbedder(), reranker=reranker, rerank_factor=10)
        for text, doc_id in DOCS:
            engine.add_text(text, id=doc_id)
        print(f"== {label} ==")
        for unit in engine.search(QUERY):
            rerank = unit.scores.get("rerank")
            tail = f"  rerank={rerank:.2f}" if rerank is not None else ""
            print(f"  ({unit.doc_id:8}) rel={unit.scores.get('rel', 0.0):+.3f}{tail}")
        print()


if __name__ == "__main__":
    main()
