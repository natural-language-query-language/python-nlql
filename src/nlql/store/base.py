"""Store protocol and capability declaration.

A store owns units, their vector index, and the parent documents. The filter passed to
``ann_search`` / ``scan`` is **IR data** (a WHERE sub-expression), not a compiled Python
predicate — so each adapter translates it to its own backend natively (``LocalStore`` to a
numpy mask, ``QdrantStore`` to a Qdrant ``Filter``, …). The Planner decides *which* part of
the WHERE is pushed here based on :class:`StoreCaps`; the rest runs in-memory.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from nlql.ir.nodes import Expr
from nlql.model import Document, Unit


@dataclass(frozen=True, slots=True)
class StoreCaps:
    """What a store can do natively."""

    name: str = "local"
    vector_search: bool = True
    exact: bool = True  # exact (flat) vs approximate (ANN) recall
    metadata_pushdown: bool = False  # can it filter metadata in its own query engine?
    text_pushdown: bool = False  # can it push CONTAINS(content, …) natively (e.g. SQL ILIKE)?


@runtime_checkable
class Store(Protocol):
    """The storage + retrieval interface the executor runs against."""

    def upsert(self, units: Sequence[Unit]) -> None: ...

    def add_documents(self, documents: Iterable[Document]) -> None: ...

    def get_document(self, doc_id: str) -> Document | None: ...

    def ann_search(
        self,
        vector: np.ndarray,
        k: int | None = None,
        *,
        filter: Expr | None = None,
    ) -> list[tuple[Unit, float]]:
        """Return up to ``k`` ``(unit, cosine)`` pairs matching ``filter``, best first."""
        ...

    def scan(self, filter: Expr | None = None) -> list[Unit]:
        """Return all units matching ``filter`` (non-vector path)."""
        ...

    def all_units(self) -> list[Unit]: ...

    def neighbors(self, doc_id: str, ordinal: int, window: int) -> list[Unit]:
        """Units of the same document within ``±window`` ordinals, ordinal-ordered."""
        ...

    def capabilities(self) -> StoreCaps: ...

    def __len__(self) -> int: ...
