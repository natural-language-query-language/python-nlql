"""ChromaStore — a vector store backed by Chroma, with native metadata-filter pushdown.

Declares ``metadata_pushdown=True``: the Planner pushes metadata filters here, and
:func:`to_chroma_where` translates the pushed IR into a native Chroma ``where`` dict. Runs
fully locally via Chroma's in-memory (``EphemeralClient``) mode — no server. Optional
dependency — ``pip install python-nlql[chroma]``.

Chroma has no ``$not`` operator, so negations are pushed down to the leaves via De Morgan
(``NOT (a AND b)`` → ``(NOT a) OR (NOT b)``) with operators inverted — keeping translation
total for what we push.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any, cast

from nlql.errors import NLQLError
from nlql.ir.nodes import And, Compare, Expr, Not, Or
from nlql.model import Unit
from nlql.model.vector import normalize
from nlql.plan.pushdown import normalized_compare
from nlql.store.base import StoreCaps
from nlql.store.common import BaseUnitStore
from nlql.store.filter import matches_filter

try:
    import chromadb
except ImportError:  # pragma: no cover - exercised only without the extra
    chromadb = None  # type: ignore[assignment]

_CHROMA_OP = {"==": "$eq", "!=": "$ne", "<": "$lt", ">": "$gt", "<=": "$lte", ">=": "$gte"}
_NEG_OP = {"==": "!=", "!=": "==", "<": ">=", ">": "<=", "<=": ">", ">=": "<"}


def _negate(expr: Expr) -> Expr:
    if isinstance(expr, Compare):
        return Compare(_NEG_OP[expr.op], expr.left, expr.right)
    if isinstance(expr, And):
        return Or([_negate(o) for o in expr.operands])
    if isinstance(expr, Or):
        return And([_negate(o) for o in expr.operands])
    if isinstance(expr, Not):
        return expr.operand
    raise ValueError(f"cannot negate {type(expr).__name__}")


def to_chroma_where(expr: Expr) -> dict[str, Any]:
    """Translate a pushed metadata filter (IR) into a native Chroma ``where`` dict."""
    if isinstance(expr, Compare):
        field, op, value = normalized_compare(expr)
        return {field: {_CHROMA_OP[op]: value}}
    if isinstance(expr, And):
        return {"$and": [to_chroma_where(o) for o in expr.operands]}
    if isinstance(expr, Or):
        return {"$or": [to_chroma_where(o) for o in expr.operands]}
    if isinstance(expr, Not):
        return to_chroma_where(_negate(expr.operand))
    raise ValueError(f"cannot translate {type(expr).__name__} to a Chroma where")


def _scalar_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Chroma metadata values must be scalar and non-null."""
    return {k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool))}


class ChromaStore(BaseUnitStore):
    """Chroma-backed store; pushes metadata filters into Chroma natively."""

    def __init__(self, *, client: Any = None, collection: str | None = None) -> None:
        if chromadb is None:
            raise NLQLError("ChromaStore requires the 'chroma' extra: pip install python-nlql[chroma]")
        super().__init__()
        self._client = client if client is not None else chromadb.EphemeralClient()
        # Ephemeral clients share state by settings, so default to a unique collection to
        # avoid cross-instance bleed; pass an explicit name to attach to persisted data.
        name = collection or f"nlql-{uuid.uuid4().hex[:12]}"
        self._collection = self._client.get_or_create_collection(
            name, metadata={"hnsw:space": "cosine"}
        )

    def _ensure_index(self) -> None:
        pass  # Chroma maintains its own index

    def upsert(self, units: Sequence[Unit]) -> None:
        super().upsert(units)  # validates vectors + dim, stores units, updates doc map
        ids, embeddings, metadatas, documents = [], [], [], []
        for unit in units:
            assert unit.vector is not None  # guaranteed by super().upsert
            ids.append(unit.id)
            embeddings.append(normalize(unit.vector).tolist())
            metadatas.append(_scalar_metadata(unit.metadata) or {"_nlql": 1})
            documents.append(unit.content)
        if ids:
            self._collection.upsert(
                ids=ids, embeddings=embeddings, metadatas=cast(Any, metadatas), documents=documents
            )

    def ann_search(
        self,
        vector: Any,
        k: int | None = None,
        *,
        filter: Expr | None = None,
    ) -> list[tuple[Unit, float]]:
        count = len(self._units)
        if count == 0:
            return []
        where = to_chroma_where(filter) if filter is not None else None
        n = min(k, count) if k is not None else count
        result: Any = self._collection.query(
            query_embeddings=[normalize(vector).tolist()], n_results=n, where=where
        )
        ids = result["ids"][0]
        distances = result["distances"][0]
        hits: list[tuple[Unit, float]] = []
        for uid, distance in zip(ids, distances, strict=True):
            unit = self._units[uid]
            if filter is not None and not matches_filter(filter, unit.metadata):
                continue  # defensive: identical results to LocalStore
            hits.append((unit, 1.0 - float(distance)))  # cosine similarity = 1 - cosine distance
        return hits

    def capabilities(self) -> StoreCaps:
        return StoreCaps(name="chroma", vector_search=True, exact=False, metadata_pushdown=True)
