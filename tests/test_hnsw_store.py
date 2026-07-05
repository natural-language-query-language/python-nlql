"""Tests for HnswStore (skipped when hnswlib is not installed)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("hnswlib")

from nlql.lang import parse  # noqa: E402
from nlql.model import Payload, Unit  # noqa: E402
from nlql.model.vector import normalize  # noqa: E402
from nlql.store.hnsw_store import HnswStore  # noqa: E402


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


def test_ranks_by_cosine() -> None:
    store = HnswStore()
    store.upsert(
        [
            _unit("x", "d1", 0, [1.0, 0.0]),
            _unit("y", "d2", 0, [0.0, 1.0]),
            _unit("z", "d3", 0, [0.9, 0.1]),
        ]
    )
    hits = store.ann_search(np.array([1.0, 0.0], dtype=np.float32))
    assert [u.id for u, _ in hits] == ["x", "z", "y"]
    assert hits[0][1] == pytest.approx(1.0, abs=1e-4)


def test_k_limits_results() -> None:
    store = HnswStore()
    store.upsert([_unit(f"u{i}", "d1", i, [1.0, i * 0.1]) for i in range(5)])
    assert len(store.ann_search(np.array([1.0, 0.0], dtype=np.float32), k=2)) == 2


def test_metadata_filter() -> None:
    store = HnswStore()
    store.upsert(
        [
            _unit("pub", "d1", 0, [1.0, 0.0], status="published"),
            _unit("draft", "d2", 0, [1.0, 0.0], status="draft"),
        ]
    )
    hits = store.ann_search(
        np.array([1.0, 0.0], dtype=np.float32),
        filter=parse('SELECT CHUNK WHERE meta.status == "published"').where,
    )
    assert [u.id for u, _ in hits] == ["pub"]


def test_capabilities() -> None:
    caps = HnswStore().capabilities()
    assert caps.name == "hnsw"
    assert caps.metadata_pushdown is True
    assert caps.exact is False  # approximate index


def test_dim_mismatch_raises() -> None:
    from nlql.errors import NLQLExecutionError

    store = HnswStore()
    store.upsert([_unit("a", "d1", 0, [1.0, 0.0])])
    with pytest.raises(NLQLExecutionError):
        store.ann_search(np.array([1.0, 0.0, 0.0], dtype=np.float32))
