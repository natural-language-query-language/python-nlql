"""HnswStore — sub-linear semantic recall via hnswlib (the M6 scale lever).

An approximate HNSW index over unit-normalized vectors (cosine space), so semantic recall is
sub-linear at million-scale — a drop-in for `LocalStore` behind the same `Store` protocol.
Metadata filters are applied by over-fetching from the index and filtering the (small)
candidate set in memory, which keeps results identical to the exact stores. With ``ef`` at
least the collection size the search is exact, so small collections match byte-for-byte.
Optional dependency — ``pip install python-nlql[hnsw]``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nlql.errors import NLQLError, NLQLExecutionError
from nlql.ir.nodes import Expr
from nlql.model import Unit
from nlql.model.vector import normalize
from nlql.store.base import StoreCaps
from nlql.store.common import BaseUnitStore
from nlql.store.filter import matches_filter

try:
    import hnswlib
except ImportError:  # pragma: no cover - exercised only without the extra
    hnswlib = None  # type: ignore[assignment]


class HnswStore(BaseUnitStore):
    """hnswlib-backed approximate vector store (cosine)."""

    _OVERFETCH = 10

    def __init__(
        self, *, M: int = 16, ef_construction: int = 200, ef: int | None = None
    ) -> None:
        if hnswlib is None:
            raise NLQLError("HnswStore requires the 'hnsw' extra: pip install python-nlql[hnsw]")
        super().__init__()
        self._M = M
        self._ef_construction = ef_construction
        self._ef = ef
        self._index: Any = None
        self._ids: list[str] = []

    def _ensure_index(self) -> None:
        if not self._dirty:
            return
        ids: list[str] = []
        rows: list[np.ndarray] = []
        for uid, unit in self._units.items():
            if unit.vector is None:
                continue
            ids.append(uid)
            rows.append(normalize(unit.vector))
        self._ids = ids
        if rows and self._dim:
            index = hnswlib.Index(space="cosine", dim=self._dim)
            index.init_index(
                max_elements=len(rows), ef_construction=self._ef_construction, M=self._M
            )
            index.add_items(np.stack(rows).astype(np.float32), np.arange(len(rows)))
            self._index = index
        else:
            self._index = None
        self._dirty = False

    def ann_search(
        self,
        vector: np.ndarray,
        k: int | None = None,
        *,
        filter: Expr | None = None,
    ) -> list[tuple[Unit, float]]:
        self._ensure_index()
        n = len(self._ids)
        if self._index is None or n == 0:
            return []
        query = normalize(vector).astype(np.float32).reshape(1, -1)
        if query.shape[1] != self._dim:
            raise NLQLExecutionError(f"query dim {query.shape[1]} != index dim {self._dim}")

        base = k if k is not None else n
        # Over-fetch when filtering so enough candidates survive the in-memory filter.
        topk = min(base * self._OVERFETCH, n) if filter is not None else min(base, n)
        topk = max(1, topk)
        self._index.set_ef(max(self._ef or 0, topk))
        labels, distances = self._index.knn_query(query, k=topk)

        hits: list[tuple[Unit, float]] = []
        for label, distance in zip(labels[0], distances[0], strict=True):
            unit = self._units[self._ids[int(label)]]
            if filter is not None and not matches_filter(filter, unit.metadata):
                continue
            hits.append((unit, 1.0 - float(distance)))  # cosine sim = 1 - cosine distance
        return hits[:k] if k is not None else hits

    def capabilities(self) -> StoreCaps:
        return StoreCaps(name="hnsw", vector_search=True, exact=False, metadata_pushdown=True)
