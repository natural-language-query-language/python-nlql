"""Tests for PgVectorStore: SQL translation (always) + a live e2e (needs NLQL_PG_DSN)."""

from __future__ import annotations

import os

import pytest

from nlql.lang import parse
from nlql.store.pgvector_store import to_sql_where


def _flt(nlql: str):
    return parse(nlql).where


class TestToSqlWhere:
    def test_string_equality(self) -> None:
        sql, params = to_sql_where(_flt('SELECT CHUNK WHERE meta.status == "published"'))
        assert sql == "metadata->>'status' = %s"
        assert params == ["published"]

    def test_numeric_range(self) -> None:
        sql, params = to_sql_where(_flt("SELECT CHUNK WHERE meta.year >= 2024"))
        assert sql == "(metadata->>'year')::numeric >= %s"
        assert params == [2024]

    def test_inequality(self) -> None:
        sql, _ = to_sql_where(_flt('SELECT CHUNK WHERE meta.status != "draft"'))
        assert sql == "metadata->>'status' <> %s"

    def test_contains_becomes_ilike(self) -> None:
        sql, params = to_sql_where(_flt('SELECT CHUNK WHERE content CONTAINS "agent"'))
        assert sql == "content ILIKE %s"
        assert params == ["%agent%"]

    def test_contains_escapes_wildcards(self) -> None:
        _, params = to_sql_where(_flt('SELECT CHUNK WHERE content CONTAINS "50%_off"'))
        assert params == ["%50\\%\\_off%"]  # % and _ escaped so they match literally

    def test_and_mixes_metadata_and_contains(self) -> None:
        sql, params = to_sql_where(
            _flt('SELECT CHUNK WHERE meta.status == "published" AND content CONTAINS "ai"')
        )
        assert sql == "(metadata->>'status' = %s AND content ILIKE %s)"
        assert params == ["published", "%ai%"]

    def test_or(self) -> None:
        sql, _ = to_sql_where(_flt('SELECT CHUNK WHERE meta.a == "x" OR meta.b == "y"'))
        assert sql == "(metadata->>'a' = %s OR metadata->>'b' = %s)"

    def test_not(self) -> None:
        sql, _ = to_sql_where(_flt('SELECT CHUNK WHERE NOT content CONTAINS "spam"'))
        assert sql == "NOT (content ILIKE %s)"


@pytest.mark.skipif(not os.environ.get("NLQL_PG_DSN"), reason="set NLQL_PG_DSN for the live test")
def test_live_pgvector_pushdown() -> None:
    from nlql import Document, Engine
    from nlql.embed import FakeEmbedder
    from nlql.store.pgvector_store import PgVectorStore

    store = PgVectorStore(os.environ["NLQL_PG_DSN"], table="nlql_test")
    engine = Engine(FakeEmbedder(dim=64), store=store, granularity="chunk")
    engine.add_documents(
        [
            Document.from_text("AI agents plan and act.", id="a", metadata={"status": "published"}),
            Document.from_text("Banana bread recipe.", id="b", metadata={"status": "published"}),
        ]
    )
    hits = engine.search(
        'SELECT CHUNK LET rel = SIMILARITY(content, "agents") '
        'WHERE meta.status == "published" AND content CONTAINS "agent" ORDER BY rel DESC'
    )
    assert [u.doc_id for u in hits] == ["a"]  # CONTAINS "agent" pushed as ILIKE, excludes b
