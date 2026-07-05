"""PgVectorStore — a vector store backed by Postgres + pgvector, with SQL pushdown.

Declares ``metadata_pushdown=True`` **and** ``text_pushdown=True``: the Planner pushes both
metadata comparisons and ``CONTAINS(content, …)`` here, and :func:`to_sql_where` translates
them into a parameterized SQL ``WHERE`` — metadata via ``metadata->>'field'`` (jsonb),
``CONTAINS`` via ``ILIKE`` (case-insensitive substring == the CONTAINS semantics, so pushing
is exact and results stay identical across stores). Values are always bound as parameters.

Needs a running Postgres with the ``vector`` extension. Optional dependency —
``pip install python-nlql[pgvector]``; the live path is skipped unless a DSN is provided.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from nlql.errors import NLQLError
from nlql.ir.nodes import And, Call, Compare, Expr, Literal, Not, Or
from nlql.model import Unit
from nlql.model.vector import normalize
from nlql.plan.pushdown import is_content_contains, normalized_compare
from nlql.store.base import StoreCaps
from nlql.store.common import BaseUnitStore

try:
    import psycopg
    from pgvector.psycopg import register_vector
except ImportError:  # pragma: no cover - exercised only without the extra
    psycopg = None  # type: ignore[assignment]
    register_vector = None  # type: ignore[assignment]

_SQL_OP = {"==": "=", "!=": "<>", "<": "<", ">": ">", "<=": "<=", ">=": ">="}
_SAFE_FIELD = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


def _ilike_param(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def to_sql_where(expr: Expr) -> tuple[str, list[Any]]:
    """Translate a pushed filter (IR) into a parameterized SQL ``WHERE`` fragment."""
    if isinstance(expr, Compare):
        field, op, value = normalized_compare(expr)
        if not _SAFE_FIELD.match(field):
            raise ValueError(f"unsafe metadata field {field!r}")
        sql_op = _SQL_OP[op]
        if isinstance(value, bool):
            return f"(metadata->>'{field}')::boolean {sql_op} %s", [value]
        if isinstance(value, (int, float)):
            return f"(metadata->>'{field}')::numeric {sql_op} %s", [value]
        return f"metadata->>'{field}' {sql_op} %s", [value]
    if is_content_contains(expr):
        assert isinstance(expr, Call)
        literal = expr.args[1]
        assert isinstance(literal, Literal)
        return "content ILIKE %s", [_ilike_param(str(literal.value))]
    if isinstance(expr, And):
        parts = [to_sql_where(o) for o in expr.operands]
        return "(" + " AND ".join(p[0] for p in parts) + ")", [v for p in parts for v in p[1]]
    if isinstance(expr, Or):
        parts = [to_sql_where(o) for o in expr.operands]
        return "(" + " OR ".join(p[0] for p in parts) + ")", [v for p in parts for v in p[1]]
    if isinstance(expr, Not):
        clause, params = to_sql_where(expr.operand)
        return f"NOT ({clause})", params
    raise ValueError(f"cannot translate {type(expr).__name__} to SQL")


class PgVectorStore(BaseUnitStore):
    """Postgres + pgvector store; pushes metadata and CONTAINS filters into SQL."""

    def __init__(self, dsn: str, *, table: str = "nlql_units", dim: int | None = None) -> None:
        if psycopg is None:
            raise NLQLError("PgVectorStore requires the 'pgvector' extra: pip install python-nlql[pgvector]")
        if not _SAFE_FIELD.match(table):
            raise NLQLError(f"unsafe table name {table!r}")
        super().__init__()
        self._table = table
        self._conn = psycopg.connect(dsn, autocommit=True)
        register_vector(self._conn)
        self._created = False
        self._declared_dim = dim

    def _ensure_index(self) -> None:
        pass  # Postgres maintains its own index

    def _ensure_table(self) -> None:
        if self._created:
            return
        dim = self._declared_dim or self._dim
        if dim is None:
            return
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self._conn.execute(
            f"CREATE TABLE IF NOT EXISTS {self._table} ("
            "id text PRIMARY KEY, doc_id text, content text, "
            f"metadata jsonb, embedding vector({dim}))"
        )
        self._created = True

    def upsert(self, units: Sequence[Unit]) -> None:
        super().upsert(units)  # validates vectors + dim, stores units, updates doc map
        self._ensure_table()
        import json

        with self._conn.cursor() as cur:
            for unit in units:
                assert unit.vector is not None
                cur.execute(
                    f"INSERT INTO {self._table} (id, doc_id, content, metadata, embedding) "
                    "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (id) DO UPDATE SET "
                    "doc_id=EXCLUDED.doc_id, content=EXCLUDED.content, "
                    "metadata=EXCLUDED.metadata, embedding=EXCLUDED.embedding",
                    (unit.id, unit.doc_id, unit.content, json.dumps(unit.metadata),
                     normalize(unit.vector)),
                )

    def ann_search(
        self,
        vector: Any,
        k: int | None = None,
        *,
        filter: Expr | None = None,
    ) -> list[tuple[Unit, float]]:
        if not self._units:
            return []
        query_vec = normalize(vector)
        where_sql = ""
        params: list[Any] = []
        if filter is not None:
            where_sql, params = to_sql_where(filter)
            where_sql = " WHERE " + where_sql
        limit = k if k is not None else len(self._units)
        sql = (
            f"SELECT id, embedding <=> %s AS dist FROM {self._table}{where_sql} "
            "ORDER BY dist LIMIT %s"
        )
        with self._conn.cursor() as cur:
            cur.execute(sql, [query_vec, *params, limit])
            rows = cur.fetchall()
        return [(self._units[uid], 1.0 - float(dist)) for uid, dist in rows if uid in self._units]

    def capabilities(self) -> StoreCaps:
        return StoreCaps(
            name="pgvector",
            vector_search=True,
            exact=False,
            metadata_pushdown=True,
            text_pushdown=True,
        )

    def close(self) -> None:
        self._conn.close()
