"""LocalStore — an in-process store with an exact numpy flat vector index.

Vectors are unit-normalized, so ``matrix @ query`` yields cosine similarity directly.
The flat index is O(N) per search but fully vectorized and never re-embeds. It declares
``metadata_pushdown=True``, so the Planner pushes metadata filters here (applied as a numpy
mask); hnswlib / faiss backends can slot in behind the same protocol for sub-linear recall.
"""

from __future__ import annotations

import numpy as np

from nlql.errors import NLQLExecutionError
from nlql.ir.nodes import Expr
from nlql.model import Unit
from nlql.model.vector import normalize
from nlql.store.base import StoreCaps
from nlql.store.columns import MetadataColumns
from nlql.store.common import BaseUnitStore


class LocalStore(BaseUnitStore):
    """Process-local unit store with an exact flat vector index."""

    def __init__(self) -> None:
        super().__init__()
        self._ids: list[str] = []
        self._matrix: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self._columns = MetadataColumns()

    def _ensure_index(self) -> None:
        if not self._dirty:
            return
        ids: list[str] = []
        rows: list[np.ndarray] = []
        for uid, unit in self._units.items():
            if unit.vector is None:
                continue
            ids.append(uid)
            rows.append(unit.vector)
        self._ids = ids
        self._matrix = (
            np.stack(rows).astype(np.float32)
            if rows
            else np.empty((0, self._dim or 0), dtype=np.float32)
        )
        self._columns.reset(self._ids, self._units)
        self._dirty = False

    def ann_search(
        self,
        vector: np.ndarray,
        k: int | None = None,
        *,
        filter: Expr | None = None,
    ) -> list[tuple[Unit, float]]:
        self._ensure_index()
        n = self._matrix.shape[0]
        if n == 0:
            return []
        query = normalize(vector)
        if query.shape[0] != self._matrix.shape[1]:
            raise NLQLExecutionError(
                f"query dim {query.shape[0]} != index dim {self._matrix.shape[1]}"
            )
        scores = self._matrix @ query  # cosine, since both sides are unit-normalized

        if filter is not None:
            candidate_idx = np.nonzero(self._columns.mask(filter))[0]
        else:
            candidate_idx = np.arange(n)
        if candidate_idx.size == 0:
            return []

        cand_scores = scores[candidate_idx]
        order = np.argsort(-cand_scores)
        if k is not None:
            order = order[:k]
        return [
            (self._units[self._ids[int(candidate_idx[o])]], float(cand_scores[o])) for o in order
        ]

    def scan(self, filter: Expr | None = None) -> list[Unit]:
        if filter is None:
            return list(self._units.values())
        self._ensure_index()
        keep = self._columns.mask(filter)
        return [self._units[self._ids[i]] for i in np.nonzero(keep)[0]]

    def capabilities(self) -> StoreCaps:
        return StoreCaps(name="local", vector_search=True, exact=True, metadata_pushdown=True)
