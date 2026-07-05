"""Query plan data structures.

``score_key`` is the shared canonical identity of a scoring call. The Planner uses it to
dedup scoring work; the Executor's evaluator uses it to read the precomputed value back
from ``Unit.scores``. Because both derive the key from the same call, no side-channel map
is needed — the "no special case" property falls out of a stable key.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from nlql.ir.nodes import Call, Expr, Query


def score_key(call: Call) -> str:
    """Canonical, stable identity of a scoring call (e.g. a SIMILARITY invocation)."""
    return json.dumps(call.to_dict(), sort_keys=True, ensure_ascii=False)


@dataclass(frozen=True, slots=True)
class Scorer:
    """A semantic score to compute during recall."""

    key: str
    query_text: str
    path: str
    call: Call
    vector_name: str = "default"  # which named unit vector to score against


@dataclass
class QueryPlan:
    """The analyzed form of a query, ready for the executor."""

    query: Query
    scorers: list[Scorer] = field(default_factory=list)
    bindings: dict[str, Expr] = field(default_factory=dict)
    granularity: str = "sentence"
    pushed_filter: Expr | None = None  # translated to the store's native filter
    residual_filter: Expr | None = None  # evaluated in-memory on returned candidates
    store: str = "local"

    def explain(self) -> dict[str, Any]:
        """A JSON-friendly description of the plan for EXPLAIN."""
        select = self.query.select
        return {
            "select": {"unit": select.unit, "window": select.window},
            "granularity": self.granularity,
            "store": self.store,
            "scores": [
                {"key_alias": self._alias_for(s), "query": s.query_text, "path": s.path}
                for s in self.scorers
            ],
            "bindings": list(self.bindings),
            "filter": {
                "pushed": self.pushed_filter.to_dict() if self.pushed_filter is not None else None,
                "residual": (
                    self.residual_filter.to_dict() if self.residual_filter is not None else None
                ),
            },
            "order_by": [
                {"expr": k.expr.to_dict(), "desc": k.desc} for k in self.query.order_by
            ],
            "limit": self.query.limit,
            "recall": "exact-flat" if self.scorers else "scan",
        }

    def _alias_for(self, scorer: Scorer) -> str | None:
        for name, expr in self.bindings.items():
            if isinstance(expr, Call) and score_key(expr) == scorer.key:
                return name
        return None
