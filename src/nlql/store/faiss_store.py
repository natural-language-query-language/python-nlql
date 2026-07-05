"""FaissStore — an exact vector store backed by faiss (``IndexFlatIP``).

A pure vector index: it declares ``metadata_pushdown=False``, so the Planner keeps every
filter residual (evaluated in-memory on the returned candidates). This exercises the
"vector recall at the store, filter in memory" half of the pushdown design. Optional
dependency — ``pip install python-nlql[faiss]``.
"""

from __future__ import annotations

import numpy as np

from nlql.errors import NLQLError, NLQLExecutionError
from nlql.ir.nodes import Expr
from nlql.model import Unit
from nlql.model.vector import normalize
from nlql.store.base import StoreCaps
from nlql.store.common import BaseUnitStore
from nlql.store.filter import matches_filter

try:
    import faiss
except ImportError:  # pragma: no cover - exercised only without the extra
    faiss = None  # type: ignore[assignment]


class FaissStore(BaseUnitStore):
    """Flat inner-product faiss index over unit-normalized vectors (cosine)."""

    def __init__(self) -> None:
        if faiss is None:
            raise NLQLError("FaissStore requires the 'faiss' extra: pip install python-nlql[faiss]")
        super().__init__()
        self._ids: list[str] = []
        self._index: object | None = None

    def _ensure_index(self) -> None:
        if not self._dirty:
            return
        assert faiss is not None
        ids: list[str] = []
        rows: list[np.ndarray] = []
        for uid, unit in self._units.items():
            if unit.vector is None:
                continue
            ids.append(uid)
            rows.append(unit.vector)
        self._ids = ids
        if rows and self._dim:
            index = faiss.IndexFlatIP(self._dim)
            index.add(np.stack(rows).astype(np.float32))
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
        if self._index is None or not self._ids:
            return []
        query = normalize(vector).astype(np.float32).reshape(1, -1)
        if query.shape[1] != self._dim:
            raise NLQLExecutionError(f"query dim {query.shape[1]} != index dim {self._dim}")
        n = len(self._ids)
        # Over-search when a residual filter must be applied post-hoc (defensive: the
        # Planner will not push a filter to a no-pushdown store, but stay correct anyway).
        search_k = n if (k is None or filter is not None) else min(k, n)
        scores, idxs = self._index.search(query, search_k)  # type: ignore[attr-defined]

        hits: list[tuple[Unit, float]] = []
        for score, i in zip(scores[0], idxs[0], strict=True):
            if i < 0:  # faiss pads with -1 when fewer than search_k results exist
                continue
            unit = self._units[self._ids[int(i)]]
            if filter is not None and not matches_filter(filter, unit.metadata):
                continue
            hits.append((unit, float(score)))
            if k is not None and len(hits) >= k:
                break
        return hits

    def capabilities(self) -> StoreCaps:
        return StoreCaps(name="faiss", vector_search=True, exact=True, metadata_pushdown=False)
