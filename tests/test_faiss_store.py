"""Tests for FaissStore (skipped when faiss is not installed)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("faiss")

from nlql.lang import parse  # noqa: E402
from nlql.model import Payload, Unit  # noqa: E402
from nlql.model.vector import normalize  # noqa: E402
from nlql.store.faiss_store import FaissStore  # noqa: E402


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
    store = FaissStore()
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


def test_k_limits_results() -> None:
    store = FaissStore()
    store.upsert([_unit(f"u{i}", "d1", i, [1.0, i * 0.1]) for i in range(5)])
    assert len(store.ann_search(np.array([1.0, 0.0], dtype=np.float32), k=2)) == 2


def test_no_metadata_pushdown() -> None:
    caps = FaissStore().capabilities()
    assert caps.name == "faiss"
    assert caps.exact is True
    assert caps.metadata_pushdown is False  # filters stay residual


def test_defensive_filter_still_correct() -> None:
    # Even though the Planner won't push here, a passed filter must be honored.
    store = FaissStore()
    store.upsert(
        [
            _unit("pub", "d1", 0, [1.0, 0.0], status="published"),
            _unit("draft", "d1", 1, [1.0, 0.0], status="draft"),
        ]
    )
    hits = store.ann_search(
        np.array([1.0, 0.0], dtype=np.float32),
        filter=parse('SELECT CHUNK WHERE meta.status == "published"').where,
    )
    assert [u.id for u, _ in hits] == ["pub"]


def test_inherits_scan_and_neighbors() -> None:
    store = FaissStore()
    store.upsert([_unit(f"s{i}", "d1", i, [1.0, i * 0.01], year=2024 - i) for i in range(4)])
    assert [u.ordinal for u in store.neighbors("d1", 1, 1)] == [0, 1, 2]
    kept = store.scan(parse("SELECT CHUNK WHERE meta.year >= 2023").where)
    assert {u.id for u in kept} == {"s0", "s1"}
