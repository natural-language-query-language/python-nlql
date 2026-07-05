"""Tests for ChromaStore and IR→Chroma where translation (skipped without chromadb)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("chromadb")

from nlql.lang import parse  # noqa: E402
from nlql.model import Payload, Unit  # noqa: E402
from nlql.model.vector import normalize  # noqa: E402
from nlql.store.chroma_store import ChromaStore, to_chroma_where  # noqa: E402


def _flt(nlql: str):
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


class TestTranslation:
    def test_equality(self) -> None:
        assert to_chroma_where(_flt('SELECT CHUNK WHERE meta.status == "published"')) == {
            "status": {"$eq": "published"}
        }

    def test_range(self) -> None:
        assert to_chroma_where(_flt("SELECT CHUNK WHERE meta.year >= 2024")) == {
            "year": {"$gte": 2024}
        }

    def test_and(self) -> None:
        where = to_chroma_where(_flt('SELECT CHUNK WHERE meta.a == "x" AND meta.b >= 3'))
        assert where == {"$and": [{"a": {"$eq": "x"}}, {"b": {"$gte": 3}}]}

    def test_or(self) -> None:
        where = to_chroma_where(_flt('SELECT CHUNK WHERE meta.a == "x" OR meta.b == "y"'))
        assert where == {"$or": [{"a": {"$eq": "x"}}, {"b": {"$eq": "y"}}]}

    def test_not_uses_de_morgan(self) -> None:
        # NOT (status == draft)  ->  status != draft
        assert to_chroma_where(_flt('SELECT CHUNK WHERE NOT meta.status == "draft"')) == {
            "status": {"$ne": "draft"}
        }


class TestEndToEnd:
    def test_native_filter_pushdown(self) -> None:
        store = ChromaStore()
        store.upsert(
            [
                _unit("pub", "d1", 0, [1.0, 0.0], status="published", year=2024),
                _unit("draft", "d2", 0, [1.0, 0.0], status="draft", year=2024),
                _unit("old", "d3", 0, [1.0, 0.0], status="published", year=2020),
            ]
        )
        hits = store.ann_search(
            np.array([1.0, 0.0], dtype=np.float32),
            filter=_flt('SELECT CHUNK WHERE meta.status == "published" AND meta.year >= 2024'),
        )
        assert [u.id for u, _ in hits] == ["pub"]

    def test_ranks_by_cosine(self) -> None:
        store = ChromaStore()
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

    def test_capabilities(self) -> None:
        caps = ChromaStore().capabilities()
        assert caps.name == "chroma"
        assert caps.metadata_pushdown is True
