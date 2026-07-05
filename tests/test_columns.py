"""Tests that the vectorized column mask matches per-row matches_filter exactly."""

from __future__ import annotations

import numpy as np
import pytest

from nlql.lang import parse
from nlql.model import Payload, Unit
from nlql.store.columns import MetadataColumns
from nlql.store.filter import matches_filter

METAS = [
    {"status": "published", "year": 2024},
    {"status": "draft", "year": 2020},
    {"status": "published", "year": 2024},
    {"status": "archived"},  # missing 'year'
    {"year": "2025"},  # 'year' as a string, missing 'status'
    {"status": "published", "date": "2024-06-01"},
]

FILTERS = [
    'meta.status == "published"',
    'meta.status != "draft"',
    "meta.year >= 2024",
    "meta.year < 2024",
    "meta.year == 2024",
    'meta.status == "published" AND meta.year >= 2024',
    'meta.status == "published" OR meta.year < 2021',
    'NOT meta.status == "draft"',
    'meta.date > "2024-01-01"',
]


def _columns():
    ids = [f"u{i}" for i in range(len(METAS))]
    units = {
        uid: Unit(id=uid, doc_id="d", kind="sentence", payload=Payload.text(uid), metadata=m)
        for uid, m in zip(ids, METAS, strict=True)
    }
    cols = MetadataColumns()
    cols.reset(ids, units)
    return cols, ids, units


def _reference(filter_expr, ids, units) -> list[bool]:
    return [matches_filter(filter_expr, units[uid].metadata) for uid in ids]


@pytest.mark.parametrize("query", FILTERS)
def test_column_mask_matches_matches_filter(query: str) -> None:
    cols, ids, units = _columns()
    where = parse(f"SELECT CHUNK WHERE {query}").where
    mask = cols.mask(where)
    assert mask.tolist() == _reference(where, ids, units)


def test_column_caching_reused() -> None:
    cols, _, _ = _columns()
    where = parse('SELECT CHUNK WHERE meta.status == "published"').where
    m1 = cols.mask(where)
    m2 = cols.mask(where)  # second call hits the cached column
    assert np.array_equal(m1, m2)


def test_unhashable_metadata_falls_back() -> None:
    ids = ["a", "b"]
    units = {
        "a": Unit(id="a", doc_id="d", kind="sentence", payload=Payload.text("a"), metadata={"tags": ["x", "y"]}),
        "b": Unit(id="b", doc_id="d", kind="sentence", payload=Payload.text("b"), metadata={"tags": "x"}),
    }
    cols = MetadataColumns()
    cols.reset(ids, units)
    mask = cols.mask(parse('SELECT CHUNK WHERE meta.tags == "x"').where)
    assert mask.tolist() == [False, True]  # list != "x", "x" == "x"
