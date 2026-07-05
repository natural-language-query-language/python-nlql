"""Hybrid engine: the same query, three store backends, identical results.

Run: python examples/hybrid_stores.py   (needs: pip install "python-nlql[faiss,qdrant]")

Demonstrates the M2 design: the Store is the abstraction and each adapter applies the
metadata filter its own way — LocalStore as a numpy mask, QdrantStore as a native Qdrant
filter, FaissStore not at all (kept residual, filtered in-memory). EXPLAIN shows the
different plans; the results are byte-identical.
"""

from __future__ import annotations

from nlql import Document, Engine
from nlql.embed import FakeEmbedder
from nlql.store import LocalStore

CORPUS = [
    Document.from_text("Machine learning models learn patterns.", id="d1", metadata={"status": "published", "year": 2024}),
    Document.from_text("Neural networks power deep learning.", id="d2", metadata={"status": "published", "year": 2025}),
    Document.from_text("Banana bread needs flour and sugar.", id="d3", metadata={"status": "draft", "year": 2024}),
    Document.from_text("Reinforcement learning trains agents.", id="d4", metadata={"status": "published", "year": 2020}),
]

QUERY = """
    SELECT SENTENCE
    LET rel = SIMILARITY(content, "deep learning networks")
    WHERE meta.status == "published" AND meta.year >= 2024
    ORDER BY rel DESC
    LIMIT 3
"""


def _backends() -> dict[str, object]:
    backends: dict[str, object] = {"LocalStore": LocalStore()}
    try:
        from nlql.store.faiss_store import FaissStore

        backends["FaissStore"] = FaissStore()
    except Exception:  # noqa: BLE001
        print("(faiss not installed — skipping)")
    try:
        from nlql.store.qdrant_store import QdrantStore

        backends["QdrantStore"] = QdrantStore()
    except Exception:  # noqa: BLE001
        print("(qdrant not installed — skipping)")
    return backends


def main() -> None:
    for name, store in _backends().items():
        engine = Engine(FakeEmbedder(dim=64), store=store)  # type: ignore[arg-type]
        engine.add_documents(CORPUS)
        plan = engine.explain(QUERY)
        pushed = plan["filter"]["pushed"] is not None
        residual = plan["filter"]["residual"] is not None
        hits = [(u.doc_id, round(u.scores["rel"], 3)) for u in engine.search(QUERY)]
        print(f"{name:12} store={plan['store']:6} pushed={pushed!s:5} residual={residual!s:5} → {hits}")


if __name__ == "__main__":
    main()
