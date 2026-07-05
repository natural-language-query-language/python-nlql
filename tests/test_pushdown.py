"""Tests for filter pushdown analysis."""

from __future__ import annotations

from nlql.lang import parse
from nlql.plan import Planner, metadata_field, split_filter
from nlql.plan.pushdown import is_content_contains, is_pushable
from nlql.registry import GLOBAL_REGISTRY
from nlql.store.base import StoreCaps

PUSH = StoreCaps(name="qdrant", metadata_pushdown=True)
NO_PUSH = StoreCaps(name="faiss", metadata_pushdown=False)
PG = StoreCaps(name="pgvector", metadata_pushdown=True, text_pushdown=True)


def where_of(nlql: str):
    return parse(nlql).where


class TestMetadataField:
    def test_meta_prefix_dropped(self) -> None:
        assert metadata_field(where_of('SELECT CHUNK WHERE meta.status == "x"').left) == "status"

    def test_bare_field(self) -> None:
        assert metadata_field(where_of('SELECT CHUNK WHERE title == "x"').left) == "title"

    def test_nested(self) -> None:
        assert metadata_field(where_of('SELECT CHUNK WHERE meta.a.b == "x"').left) == "a.b"


class TestPushability:
    def test_metadata_compare_is_pushable(self) -> None:
        assert is_pushable(where_of('SELECT CHUNK WHERE meta.status == "published"'), PUSH)

    def test_content_not_pushable(self) -> None:
        assert not is_pushable(where_of('SELECT CHUNK WHERE content CONTAINS "x"'), PUSH)

    def test_score_ref_not_pushable(self) -> None:
        w = where_of('SELECT SENTENCE LET r = SIMILARITY(content, "x") WHERE r >= 0.5')
        assert not is_pushable(w, PUSH)

    def test_and_of_metadata_is_pushable(self) -> None:
        w = where_of('SELECT CHUNK WHERE meta.status == "published" AND meta.year >= 2024')
        assert is_pushable(w, PUSH)

    def test_or_atomic_pushable(self) -> None:
        w = where_of('SELECT CHUNK WHERE meta.a == "x" OR meta.b == "y"')
        assert is_pushable(w, PUSH)

    def test_nothing_pushable_when_caps_disabled(self) -> None:
        assert not is_pushable(where_of('SELECT CHUNK WHERE meta.status == "x"'), NO_PUSH)


class TestSplit:
    def test_no_where(self) -> None:
        s = split_filter(None, PUSH)
        assert s.pushed is None and s.residual is None

    def test_all_pushed(self) -> None:
        w = where_of('SELECT CHUNK WHERE meta.status == "published" AND meta.year >= 2024')
        s = split_filter(w, PUSH)
        assert s.pushed is not None and s.residual is None

    def test_conjunctive_split(self) -> None:
        # metadata conjunct pushes; content conjunct stays residual
        w = where_of('SELECT CHUNK WHERE meta.status == "published" AND content CONTAINS "agent"')
        s = split_filter(w, PUSH)
        assert s.pushed is not None
        assert s.residual is not None
        assert s.pushed.op == "=="  # the metadata compare
        assert s.residual.name == "CONTAINS"  # the content predicate

    def test_all_residual_when_no_pushdown(self) -> None:
        w = where_of('SELECT CHUNK WHERE meta.status == "published" AND meta.year >= 2024')
        s = split_filter(w, NO_PUSH)
        assert s.pushed is None and s.residual is not None

    def test_mixed_score_and_metadata(self) -> None:
        w = where_of(
            'SELECT SENTENCE LET r = SIMILARITY(content, "x") '
            'WHERE r >= 0.5 AND meta.status == "published"'
        )
        s = split_filter(w, PUSH)
        assert s.pushed.op == "=="  # meta.status pushed
        assert s.residual is not None  # r >= 0.5 stays in memory


class TestContentContainsPushdown:
    def test_is_content_contains(self) -> None:
        assert is_content_contains(where_of('SELECT CHUNK WHERE content CONTAINS "x"'))
        assert not is_content_contains(where_of('SELECT CHUNK WHERE meta.a == "x"'))
        assert not is_content_contains(where_of('SELECT CHUNK WHERE content MATCH "x"'))

    def test_contains_pushable_only_with_text_pushdown(self) -> None:
        w = where_of('SELECT CHUNK WHERE content CONTAINS "agent"')
        assert is_pushable(w, PG)  # pgvector: text_pushdown=True
        assert not is_pushable(w, PUSH)  # qdrant: no text_pushdown → residual

    def test_mixed_meta_and_contains_all_pushed_on_pg(self) -> None:
        w = where_of('SELECT CHUNK WHERE meta.status == "published" AND content CONTAINS "ai"')
        s = split_filter(w, PG)
        assert s.pushed is not None and s.residual is None

    def test_contains_residual_on_non_text_store(self) -> None:
        w = where_of('SELECT CHUNK WHERE meta.status == "published" AND content CONTAINS "ai"')
        s = split_filter(w, PUSH)  # qdrant pushes meta, keeps CONTAINS residual
        assert s.pushed.op == "=="
        assert s.residual.name == "CONTAINS"


class TestPlannerIntegration:
    def test_plan_carries_split_and_explains(self) -> None:
        q = parse('SELECT SENTENCE LET r = SIMILARITY(content, "x") WHERE r >= 0.3 AND meta.k == "v" ORDER BY r DESC')
        plan = Planner(GLOBAL_REGISTRY.child()).plan(q, granularity="sentence", caps=PUSH)
        assert plan.pushed_filter is not None
        assert plan.residual_filter is not None
        exp = plan.explain()
        assert exp["store"] == "qdrant"
        assert exp["filter"]["pushed"] is not None
        assert exp["filter"]["residual"] is not None
