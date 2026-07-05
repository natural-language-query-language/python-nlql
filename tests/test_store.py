"""Tests for LocalStore (flat vector index, neighbors, documents)."""

from __future__ import annotations

import numpy as np
import pytest

from nlql.errors import NLQLExecutionError
from nlql.lang import parse
from nlql.model import Document, Payload, Unit
from nlql.model.vector import normalize
from nlql.store import LocalStore


def _filter(nlql: str):
    return parse(nlql).where


def _unit(uid: str, doc_id: str, ordinal: int, vec: list[float], **meta: object) -> Unit:
    return Unit(
        id=uid,
        doc_id=doc_id,
        kind="sentence",
        payload=Payload.text(uid),
        vector=normalize(vec),
        ordinal=ordinal,
        metadata=dict(meta),
    )


class TestUpsertAndScan:
    def test_len_and_all_units(self) -> None:
        store = LocalStore()
        store.upsert([_unit("a", "d1", 0, [1, 0]), _unit("b", "d1", 1, [0, 1])])
        assert len(store) == 2
        assert {u.id for u in store.all_units()} == {"a", "b"}

    def test_upsert_replaces_same_id(self) -> None:
        store = LocalStore()
        store.upsert([_unit("a", "d1", 0, [1, 0])])
        store.upsert([_unit("a", "d1", 0, [0, 1])])
        assert len(store) == 1

    def test_missing_vector_rejected(self) -> None:
        store = LocalStore()
        u = Unit(id="a", doc_id="d1", kind="sentence", payload=Payload.text("a"))
        with pytest.raises(NLQLExecutionError):
            store.upsert([u])

    def test_dim_mismatch_rejected(self) -> None:
        store = LocalStore()
        store.upsert([_unit("a", "d1", 0, [1, 0])])
        with pytest.raises(NLQLExecutionError):
            store.upsert([_unit("b", "d1", 1, [1, 0, 0])])


class TestAnnSearch:
    def test_ranks_by_cosine(self) -> None:
        store = LocalStore()
        store.upsert(
            [
                _unit("x", "d1", 0, [1.0, 0.0]),
                _unit("y", "d1", 1, [0.0, 1.0]),
                _unit("z", "d1", 2, [0.9, 0.1]),
            ]
        )
        hits = store.ann_search(np.array([1.0, 0.0], dtype=np.float32))
        assert [u.id for u, _ in hits] == ["x", "z", "y"]
        assert hits[0][1] == pytest.approx(1.0, abs=1e-5)

    def test_k_limits_results(self) -> None:
        store = LocalStore()
        store.upsert([_unit(f"u{i}", "d1", i, [1.0, i * 0.1]) for i in range(5)])
        hits = store.ann_search(np.array([1.0, 0.0], dtype=np.float32), k=2)
        assert len(hits) == 2

    def test_filter_prefilter(self) -> None:
        store = LocalStore()
        store.upsert(
            [
                _unit("pub", "d1", 0, [1.0, 0.0], status="published"),
                _unit("draft", "d1", 1, [1.0, 0.0], status="draft"),
            ]
        )
        hits = store.ann_search(
            np.array([1.0, 0.0], dtype=np.float32),
            filter=_filter('SELECT CHUNK WHERE meta.status == "published"'),
        )
        assert [u.id for u, _ in hits] == ["pub"]

    def test_empty_store(self) -> None:
        assert LocalStore().ann_search(np.array([1.0, 0.0], dtype=np.float32)) == []

    def test_query_dim_mismatch_raises(self) -> None:
        store = LocalStore()
        store.upsert([_unit("a", "d1", 0, [1, 0])])
        with pytest.raises(NLQLExecutionError):
            store.ann_search(np.array([1.0, 0.0, 0.0], dtype=np.float32))


class TestNeighborsAndDocuments:
    def test_neighbors_window(self) -> None:
        store = LocalStore()
        store.upsert([_unit(f"s{i}", "d1", i, [1.0, i * 0.01]) for i in range(5)])
        nb = store.neighbors("d1", ordinal=2, window=1)
        assert [u.ordinal for u in nb] == [1, 2, 3]

    def test_neighbors_clipped_at_edges(self) -> None:
        store = LocalStore()
        store.upsert([_unit(f"s{i}", "d1", i, [1.0, i * 0.01]) for i in range(3)])
        assert [u.ordinal for u in store.neighbors("d1", 0, 2)] == [0, 1, 2]

    def test_documents_round_trip(self) -> None:
        store = LocalStore()
        doc = Document.from_text("body", id="d1", metadata={"k": "v"})
        store.add_documents([doc])
        assert store.get_document("d1") is doc
        assert store.get_document("missing") is None

    def test_capabilities(self) -> None:
        caps = LocalStore().capabilities()
        assert caps.vector_search and caps.exact and caps.metadata_pushdown

    def test_scan_with_filter(self) -> None:
        store = LocalStore()
        store.upsert(
            [
                _unit("a", "d1", 0, [1.0, 0.0], year=2024),
                _unit("b", "d1", 1, [1.0, 0.0], year=2020),
            ]
        )
        kept = store.scan(_filter("SELECT CHUNK WHERE meta.year >= 2024"))
        assert [u.id for u in kept] == ["a"]
        assert len(store.scan()) == 2  # no filter → all
