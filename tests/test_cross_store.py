"""M2 acceptance: the same query returns identical results on every store backend.

LocalStore pushes metadata filters natively, FaissStore keeps them residual, and
QdrantStore translates them to a native Qdrant filter — yet results must be byte-identical.
Backends that are not installed are skipped.
"""

from __future__ import annotations

import pytest

from nlql import Document, Engine
from nlql.embed import FakeEmbedder
from nlql.store import LocalStore

CORPUS = [
    Document.from_text("Machine learning models learn patterns from data.", id="d1",
                       metadata={"status": "published", "year": 2024}),
    Document.from_text("Neural networks power deep learning systems.", id="d2",
                       metadata={"status": "published", "year": 2025}),
    Document.from_text("Banana bread needs flour and sugar to bake.", id="d3",
                       metadata={"status": "draft", "year": 2024}),
    Document.from_text("Reinforcement learning trains agents via rewards.", id="d4",
                       metadata={"status": "published", "year": 2020}),
]

QUERIES = [
    # semantic + pushed metadata filter + limit
    'SELECT SENTENCE LET rel = SIMILARITY(content, "deep learning networks") '
    'WHERE meta.status == "published" ORDER BY rel DESC LIMIT 3',
    # metadata-only (no vector)
    'SELECT SENTENCE WHERE meta.status == "published" AND meta.year >= 2024',
    # pushed metadata + residual score threshold
    'SELECT SENTENCE LET rel = SIMILARITY(content, "learning") '
    'WHERE rel >= -1.0 AND meta.year >= 2024 ORDER BY rel DESC',
    # document granularity
    'SELECT DOCUMENT LET rel = SIMILARITY(content, "learning agents") ORDER BY rel DESC LIMIT 2',
    # pure semantic, no filter
    'SELECT SENTENCE LET rel = SIMILARITY(content, "machine learning") ORDER BY rel DESC LIMIT 2',
]


def _available_stores() -> dict[str, object]:
    stores: dict[str, object] = {"local": LocalStore()}
    try:
        from nlql.store.faiss_store import FaissStore

        stores["faiss"] = FaissStore()
    except Exception:  # pragma: no cover - faiss not installed
        pass
    try:
        from nlql.store.qdrant_store import QdrantStore

        stores["qdrant"] = QdrantStore()
    except Exception:  # pragma: no cover - qdrant not installed
        pass
    try:
        from nlql.store.chroma_store import ChromaStore

        stores["chroma"] = ChromaStore()
    except Exception:  # pragma: no cover - chroma not installed
        pass
    try:
        from nlql.store.hnsw_store import HnswStore

        stores["hnsw"] = HnswStore(ef=200)  # ef >= collection size → exact for small data
    except Exception:  # pragma: no cover - hnswlib not installed
        pass
    return stores


def _seeded_engine(store: object) -> Engine:
    engine = Engine(FakeEmbedder(dim=64), store=store)  # type: ignore[arg-type]
    engine.add_documents(CORPUS)
    return engine


def test_results_identical_across_stores() -> None:
    stores = _available_stores()
    if len(stores) < 2:
        pytest.skip("need at least two store backends installed")
    engines = {name: _seeded_engine(store) for name, store in stores.items()}

    for query in QUERIES:
        results = {name: [u.id for u in eng.search(query)] for name, eng in engines.items()}
        reference = results["local"]
        for name, ids in results.items():
            assert ids == reference, f"{name} diverged on {query!r}: {ids} != {reference}"


def test_explain_reflects_per_store_pushdown() -> None:
    stores = _available_stores()
    query = 'SELECT SENTENCE LET rel = SIMILARITY(content, "x") WHERE meta.status == "published" ORDER BY rel DESC'

    local_plan = _seeded_engine(stores["local"]).explain(query)
    assert local_plan["store"] == "local"
    assert local_plan["filter"]["pushed"] is not None  # metadata filter pushed
    assert local_plan["filter"]["residual"] is None

    if "faiss" in stores:
        faiss_plan = _seeded_engine(stores["faiss"]).explain(query)
        assert faiss_plan["filter"]["pushed"] is None  # no pushdown → all residual
        assert faiss_plan["filter"]["residual"] is not None

    if "qdrant" in stores:
        qdrant_plan = _seeded_engine(stores["qdrant"]).explain(query)
        assert qdrant_plan["store"] == "qdrant"
        assert qdrant_plan["filter"]["pushed"] is not None
