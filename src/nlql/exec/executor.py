"""The Executor — runs a planned query against a store.

Pipeline: score candidates via the flat index (one matmul over cached vectors — never
re-embedding), filter WHERE, order, transform to the requested granularity
(SENTENCE / SPAN / DOCUMENT), and limit. Named scores are surfaced on the result units
under their LET aliases.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from nlql.embed.base import Embedder
from nlql.errors import NLQLPlanError
from nlql.exec.evaluate import Evaluator
from nlql.ir.nodes import Call, OrderKey, Query, Ref
from nlql.model import Payload, Span, Unit
from nlql.plan.plan import QueryPlan, score_key
from nlql.plan.planner import Planner
from nlql.rerank.base import Reranker
from nlql.store.base import Store, StoreCaps
from nlql.types.coerce import as_date
from nlql.types.core import TypeTag


class Executor:
    """Executes queries against a :class:`~nlql.store.base.Store`."""

    def __init__(
        self,
        store: Store,
        registry: Any,
        embedder: Embedder,
        *,
        granularity: str = "sentence",
        field_types: dict[str, TypeTag] | None = None,
        reranker: Reranker | None = None,
        rerank_factor: int = 5,
        named_embedders: dict[str, Embedder] | None = None,
    ) -> None:
        self._store = store
        self._registry = registry
        self._embedder = embedder
        self._granularity = granularity
        self._field_types = field_types or {}
        self._reranker = reranker
        self._rerank_factor = max(1, rerank_factor)
        self._named_embedders = named_embedders or {}
        self._planner = Planner(registry)
        self._evaluator = Evaluator(registry, self._field_types)

    _OVERFETCH = 10

    @property
    def reranker(self) -> Reranker | None:
        return self._reranker

    def plan(self, query: Query) -> QueryPlan:
        return self._planner.plan(
            query,
            granularity=self._granularity,
            caps=self._store.capabilities(),
            field_types=self._field_types,
        )

    def execute(
        self,
        query: Query,
        *,
        reranker: Reranker | None = None,
        rerank_query: str | None = None,
    ) -> list[Unit]:
        active = reranker if reranker is not None else self._reranker
        plan = self.plan(query)
        # A reranker refines a wider candidate set, so recall over-fetches before it.
        overfetch = self._rerank_factor if (active is not None and plan.scorers) else 1
        candidates = self._recall(query, plan, overfetch=overfetch)
        self._apply_scores(candidates, plan)
        survivors = self._filter(candidates, plan)
        ordered = self._order(survivors, plan)
        results = self._materialize(ordered, plan)
        if active is not None and plan.scorers:
            question = rerank_query if rerank_query is not None else plan.scorers[0].query_text
            results = active.rerank(question, results)
        if query.limit is not None:
            results = results[: query.limit]
        self._surface_named_scores(results, plan)
        return results

    # -- recall (pushes the store-native filter down) --------------------------
    def _recall(self, query: Query, plan: QueryPlan, *, overfetch: int = 1) -> list[Unit]:
        pushed = plan.pushed_filter
        # No scorer, or a named-vector scorer (the store's ANN index covers the default vector
        # only) → a filtered scan; named vectors are then scored in-memory in _apply_scores.
        if not plan.scorers or any(s.vector_name != "default" for s in plan.scorers):
            return self._store.scan(filter=pushed)
        primary = plan.scorers[0]
        query_vector = self._embedder.embed([primary.query_text])[0]
        k = self._recall_k(query, plan, self._store.capabilities())
        if k is not None and overfetch > 1:
            k *= overfetch
        hits = self._store.ann_search(query_vector, k=k, filter=pushed)
        return [unit for unit, _ in hits]

    def _recall_k(self, query: Query, plan: QueryPlan, caps: StoreCaps) -> int | None:
        if query.limit is None:
            return None
        # Exact top-k is only safe when one scorer drives the ordering and nothing else
        # (residual filter, extra scorers, or a different sort key) can reorder/drop hits.
        simple = (
            len(plan.scorers) == 1
            and plan.residual_filter is None
            and self._orders_by_primary_desc(query, plan)
        )
        if simple:
            return query.limit
        # Otherwise recall wider: exact stores can take everything; approximate over-fetch.
        return None if caps.exact else query.limit * self._OVERFETCH

    @staticmethod
    def _orders_by_primary_desc(query: Query, plan: QueryPlan) -> bool:
        if not query.order_by:
            return True  # default ordering is the primary score, descending
        if len(query.order_by) != 1 or not query.order_by[0].desc:
            return False
        expr = query.order_by[0].expr
        target = plan.bindings.get(expr.name) if isinstance(expr, Ref) else expr
        return isinstance(target, Call) and score_key(target) == plan.scorers[0].key

    # -- scoring ---------------------------------------------------------------
    def _apply_scores(self, units: list[Unit], plan: QueryPlan) -> None:
        if not plan.scorers or not units:
            return
        for scorer in plan.scorers:
            # Each scorer lives in its own vector space, embedded by its own embedder.
            embedder = self._named_embedders.get(scorer.vector_name, self._embedder)
            query_vector = embedder.embed([scorer.query_text])[0]
            unit_vectors = [u.get_vector(scorer.vector_name) for u in units]
            present = [v for v in unit_vectors if v is not None]
            if len(present) == len(units):
                column = np.stack(present) @ query_vector
                for unit, value in zip(units, column, strict=True):
                    unit.scores[scorer.key] = float(value)
            else:
                # Units lacking this named vector score below any cosine, so they drop out.
                for unit, vector in zip(units, unit_vectors, strict=True):
                    unit.scores[scorer.key] = float(vector @ query_vector) if vector is not None else -2.0

    # -- filtering -------------------------------------------------------------
    def _filter(self, units: list[Unit], plan: QueryPlan) -> list[Unit]:
        residual = plan.residual_filter  # pushed part already applied by the store
        if residual is None:
            return list(units)
        return [
            u
            for u in units
            if self._evaluator.truthy(self._evaluator.eval(residual, u, plan.bindings))
        ]

    # -- ordering --------------------------------------------------------------
    def _order(self, units: list[Unit], plan: QueryPlan) -> list[Unit]:
        order_by = plan.query.order_by
        result = list(units)
        if not order_by:
            if plan.scorers:  # sensible default: best match first
                key = plan.scorers[0].key
                result.sort(key=lambda u: u.scores.get(key, 0.0), reverse=True)
            return result
        # Stable multi-key sort: apply keys from least to most significant.
        for order_key in reversed(order_by):
            result.sort(key=self._sort_key(order_key, plan), reverse=order_key.desc)
        return result

    def _sort_key(self, order_key: OrderKey, plan: QueryPlan) -> Callable[[Unit], tuple[int, Any]]:
        def key(unit: Unit) -> tuple[int, Any]:
            return self._sort_proxy(self._order_value(order_key, unit, plan))

        return key

    def _order_value(self, order_key: OrderKey, unit: Unit, plan: QueryPlan) -> Any:
        return self._evaluator.eval(order_key.expr, unit, plan.bindings)

    @staticmethod
    def _sort_proxy(value: Any) -> tuple[int, Any]:
        """Map a value to a total-order-safe (rank, key) tuple; nulls sort last."""
        if value is None:
            return (3, 0.0)
        if isinstance(value, bool):
            return (0, float(value))
        if isinstance(value, (int, float)):
            return (0, float(value))
        if isinstance(value, str):
            parsed = as_date(value)
            if parsed is not None:
                return (1, parsed.timestamp())
            return (2, value)
        return (2, str(value))

    # -- granularity -----------------------------------------------------------
    def _materialize(self, units: list[Unit], plan: QueryPlan) -> list[Unit]:
        select = plan.query.select
        if select.unit == self._granularity:
            if select.window is None:
                return units
            return [self._make_span(u, select.window) for u in units]
        if select.unit == "document":
            if select.window is not None:
                raise NLQLPlanError("SPAN(DOCUMENT) is not supported")
            return self._aggregate_documents(units)
        raise NLQLPlanError(
            f"store is indexed at '{self._granularity}' granularity; cannot SELECT "
            f"'{select.unit}'. Use {self._granularity}, SPAN({self._granularity}), or document."
        )

    def _make_span(self, center: Unit, window: int) -> Unit:
        neighbors = self._store.neighbors(center.doc_id, center.ordinal, window)
        if not neighbors:
            neighbors = [center]
        text = " ".join(u.content for u in neighbors)
        span = Span(center=center.ordinal, start=neighbors[0].ordinal, end=neighbors[-1].ordinal)
        return Unit(
            id=f"{center.doc_id}#span:{center.ordinal}~{window}",
            doc_id=center.doc_id,
            kind="span",
            payload=Payload.text(text),
            metadata=dict(center.metadata),
            vector=center.vector,
            span=span,
            scores=dict(center.scores),
            ordinal=center.ordinal,
        )

    def _aggregate_documents(self, units: list[Unit]) -> list[Unit]:
        """One result per document, represented by its best-ranked matching unit."""
        results: list[Unit] = []
        seen: set[str] = set()
        for rep in units:  # already ordered; first per doc == best
            if rep.doc_id in seen:
                continue
            seen.add(rep.doc_id)
            doc = self._store.get_document(rep.doc_id)
            if doc is not None:
                text = " ".join(p.as_text for p in doc.payloads if p.is_text)
                metadata = dict(doc.metadata)
            else:
                text, metadata = rep.content, dict(rep.metadata)
            results.append(
                Unit(
                    id=rep.doc_id,
                    doc_id=rep.doc_id,
                    kind="document",
                    payload=Payload.text(text),
                    metadata=metadata,
                    vector=rep.vector,
                    scores=dict(rep.scores),
                    ordinal=0,
                )
            )
        return results

    # -- presentation ----------------------------------------------------------
    def _surface_named_scores(self, results: list[Unit], plan: QueryPlan) -> None:
        for unit in results:
            for name, expr in plan.bindings.items():
                try:
                    value = self._evaluator.eval(expr, unit, plan.bindings)
                except Exception:
                    continue
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    unit.scores[name] = float(value)
            # Drop internal score-key entries so results expose only friendly names.
            for key in [k for k in unit.scores if k.startswith("{")]:
                del unit.scores[key]
