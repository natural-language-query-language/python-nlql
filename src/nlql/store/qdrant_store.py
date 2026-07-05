"""QdrantStore — a vector store backed by Qdrant, with native metadata-filter pushdown.

Declares ``metadata_pushdown=True``: the Planner pushes metadata filters here, and
:func:`to_qdrant_filter` translates the pushed IR sub-expression into a native Qdrant
``Filter`` so Qdrant returns only matching vectors. This is the "full pushdown" half of the
design. It runs fully locally via Qdrant's in-memory mode (``location=":memory:"``), so it
needs no server. Optional dependency — ``pip install python-nlql[qdrant]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from nlql.errors import NLQLError
from nlql.ir.nodes import And, Compare, Expr, Not, Or
from nlql.model import Unit
from nlql.model.vector import normalize
from nlql.plan.pushdown import normalized_compare
from nlql.store.base import StoreCaps
from nlql.store.common import BaseUnitStore
from nlql.store.filter import matches_filter

try:
    from qdrant_client import QdrantClient
    from qdrant_client import models as qm
except ImportError:  # pragma: no cover - exercised only without the extra
    QdrantClient = None  # type: ignore[assignment,misc]
    qm = None  # type: ignore[assignment]

_RANGE_KEY = {"<": "lt", ">": "gt", "<=": "lte", ">=": "gte"}


def _condition(expr: Expr) -> Any:
    """Translate one pushed sub-expression into a Qdrant condition/filter."""
    if isinstance(expr, Compare):
        field, op, value = normalized_compare(expr)
        if op == "==":
            return qm.FieldCondition(key=field, match=qm.MatchValue(value=value))
        if op == "!=":
            return qm.Filter(
                must_not=[qm.FieldCondition(key=field, match=qm.MatchValue(value=value))]
            )
        return qm.FieldCondition(key=field, range=qm.Range(**{_RANGE_KEY[op]: value}))
    if isinstance(expr, And):
        return qm.Filter(must=[_condition(o) for o in expr.operands])
    if isinstance(expr, Or):
        return qm.Filter(should=[_condition(o) for o in expr.operands])
    if isinstance(expr, Not):
        return qm.Filter(must_not=[_condition(expr.operand)])
    raise ValueError(f"cannot translate {type(expr).__name__} to a Qdrant filter")


def to_qdrant_filter(expr: Expr) -> Any:
    """Translate a pushed metadata filter (IR) into a native Qdrant ``Filter``."""
    condition = _condition(expr)
    return condition if isinstance(condition, qm.Filter) else qm.Filter(must=[condition])


class QdrantStore(BaseUnitStore):
    """Qdrant-backed store; pushes metadata filters into Qdrant natively."""

    def __init__(
        self, *, client: Any = None, collection: str = "nlql", location: str = ":memory:"
    ) -> None:
        if QdrantClient is None:
            raise NLQLError("QdrantStore requires the 'qdrant' extra: pip install python-nlql[qdrant]")
        super().__init__()
        self._client = client if client is not None else QdrantClient(location=location)
        self._collection = collection
        self._pid_of: dict[str, int] = {}
        self._next_pid = 0

    def _pid(self, uid: str) -> int:
        if uid not in self._pid_of:
            self._pid_of[uid] = self._next_pid
            self._next_pid += 1
        return self._pid_of[uid]

    def _ensure_collection(self) -> None:
        if self._dim and not self._client.collection_exists(self._collection):
            self._client.create_collection(
                self._collection,
                vectors_config=qm.VectorParams(size=self._dim, distance=qm.Distance.COSINE),
            )

    def _ensure_index(self) -> None:
        pass  # Qdrant maintains its own index; nothing to rebuild locally

    def upsert(self, units: Sequence[Unit]) -> None:
        super().upsert(units)  # validates vectors + dim, stores units, updates doc map
        self._ensure_collection()
        points = []
        for unit in units:
            assert unit.vector is not None  # guaranteed set by super().upsert
            points.append(
                qm.PointStruct(
                    id=self._pid(unit.id),
                    vector=normalize(unit.vector).tolist(),
                    payload={**unit.metadata, "__uid": unit.id},
                )
            )
        if points:
            self._client.upsert(self._collection, points=points)

    def ann_search(
        self,
        vector: Any,
        k: int | None = None,
        *,
        filter: Expr | None = None,
    ) -> list[tuple[Unit, float]]:
        if not self._units:
            return []
        query_filter = to_qdrant_filter(filter) if filter is not None else None
        limit = k if k is not None else len(self._units)
        response = self._client.query_points(
            self._collection,
            query=normalize(vector).tolist(),
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
        hits: list[tuple[Unit, float]] = []
        for point in response.points:
            assert point.payload is not None  # every point carries its "__uid"
            unit = self._units[point.payload["__uid"]]
            # Defensive re-check: translation is total for what we push, so this never
            # drops a valid hit — it just guarantees byte-identical results to LocalStore.
            if filter is not None and not matches_filter(filter, unit.metadata):
                continue
            hits.append((unit, float(point.score)))
        return hits

    def capabilities(self) -> StoreCaps:
        return StoreCaps(name="qdrant", vector_search=True, exact=False, metadata_pushdown=True)
