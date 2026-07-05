"""Tests for QdrantStore and IR→Qdrant filter translation (skipped without qdrant)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qdrant_client")

from nlql.lang import parse  # noqa: E402
from nlql.model import Payload, Unit  # noqa: E402
from nlql.model.vector import normalize  # noqa: E402
from nlql.store.qdrant_store import QdrantStore, to_qdrant_filter  # noqa: E402


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
        f = to_qdrant_filter(_flt('SELECT CHUNK WHERE meta.status == "published"'))
        assert f.must[0].key == "status"
        assert f.must[0].match.value == "published"

    def test_numeric_range(self) -> None:
        f = to_qdrant_filter(_flt("SELECT CHUNK WHERE meta.year >= 2024"))
        assert f.must[0].range.gte == 2024

    def test_flipped_operand_order(self) -> None:
        f = to_qdrant_filter(_flt("SELECT CHUNK WHERE 2024 <= meta.year"))
        assert f.must[0].range.gte == 2024  # 2024 <= year  ==>  year >= 2024

    def test_inequality(self) -> None:
        f = to_qdrant_filter(_flt('SELECT CHUNK WHERE meta.status != "draft"'))
        assert f.must_not[0].match.value == "draft"

    def test_and_of_conditions(self) -> None:
        f = to_qdrant_filter(_flt('SELECT CHUNK WHERE meta.status == "p" AND meta.year >= 2024'))
        assert len(f.must) == 2

    def test_or_maps_to_should(self) -> None:
        f = to_qdrant_filter(_flt('SELECT CHUNK WHERE meta.a == "x" OR meta.b == "y"'))
        assert len(f.should) == 2


class TestEndToEnd:
    def test_native_filter_pushdown(self) -> None:
        store = QdrantStore()
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
        store = QdrantStore()
        store.upsert(
            [
                _unit("x", "d1", 0, [1.0, 0.0]),
                _unit("y", "d2", 0, [0.0, 1.0]),
                _unit("z", "d3", 0, [0.9, 0.1]),
            ]
        )
        hits = store.ann_search(np.array([1.0, 0.0], dtype=np.float32))
        assert [u.id for u, _ in hits] == ["x", "z", "y"]

    def test_capabilities(self) -> None:
        caps = QdrantStore().capabilities()
        assert caps.name == "qdrant"
        assert caps.metadata_pushdown is True
        assert caps.exact is False
