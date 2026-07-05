"""Shared base for in-process unit stores.

Holds the units, documents, and per-document ordinal map, plus the metadata scan — the
parts every local store shares. A concrete store only implements the vector index:
``_ensure_index``, ``ann_search``, and ``capabilities``. Adding a new local backend is
therefore ~40 lines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence

import numpy as np

from nlql.errors import NLQLExecutionError
from nlql.ir.nodes import Expr
from nlql.model import Document, Unit
from nlql.store.base import StoreCaps
from nlql.store.filter import matches_filter


class BaseUnitStore(ABC):
    """Common unit/document bookkeeping for local stores."""

    def __init__(self) -> None:
        self._units: dict[str, Unit] = {}
        self._docs: dict[str, Document] = {}
        self._by_doc: dict[str, list[str]] = {}
        self._dim: int | None = None
        self._dirty = True

    # -- writes ---------------------------------------------------------------
    def upsert(self, units: Sequence[Unit]) -> None:
        for unit in units:
            if unit.vector is None:
                raise NLQLExecutionError(f"unit {unit.id!r} has no vector; embed before upsert")
            vec = np.asarray(unit.vector, dtype=np.float32)
            if self._dim is None:
                self._dim = int(vec.shape[0])
            elif vec.shape[0] != self._dim:
                raise NLQLExecutionError(
                    f"vector dim mismatch: store is {self._dim}, unit {unit.id!r} is {vec.shape[0]}"
                )
            unit.vector = vec
            self._units[unit.id] = unit
        self._reindex_doc_map(units)
        self._dirty = True

    def add_documents(self, documents: Iterable[Document]) -> None:
        for doc in documents:
            self._docs[doc.id] = doc

    def _reindex_doc_map(self, units: Sequence[Unit]) -> None:
        for doc_id in {u.doc_id for u in units}:
            ids = [u.id for u in self._units.values() if u.doc_id == doc_id]
            ids.sort(key=lambda uid: self._units[uid].ordinal)
            self._by_doc[doc_id] = ids

    # -- reads ----------------------------------------------------------------
    def scan(self, filter: Expr | None = None) -> list[Unit]:
        if filter is None:
            return list(self._units.values())
        return [u for u in self._units.values() if matches_filter(filter, u.metadata)]

    def all_units(self) -> list[Unit]:
        return list(self._units.values())

    def neighbors(self, doc_id: str, ordinal: int, window: int) -> list[Unit]:
        ids = self._by_doc.get(doc_id, [])
        lo, hi = ordinal - window, ordinal + window
        return [self._units[uid] for uid in ids if lo <= self._units[uid].ordinal <= hi]

    def get_document(self, doc_id: str) -> Document | None:
        return self._docs.get(doc_id)

    @property
    def dim(self) -> int | None:
        return self._dim

    def __len__(self) -> int:
        return len(self._units)

    # -- vector index (backend-specific) --------------------------------------
    @abstractmethod
    def _ensure_index(self) -> None: ...

    @abstractmethod
    def ann_search(
        self, vector: np.ndarray, k: int | None = None, *, filter: Expr | None = None
    ) -> list[tuple[Unit, float]]: ...

    @abstractmethod
    def capabilities(self) -> StoreCaps: ...
