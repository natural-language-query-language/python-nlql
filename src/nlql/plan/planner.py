"""The Planner — semantic analysis of a Query IR against the registry.

It validates that every referenced function/alias exists, extracts the semantic scoring
calls (SIMILARITY) so the executor can compute them once via the index, and records the
alias→expr bindings. External-store pushdown splitting lands here in M2; for M1 the plan
is always fully local.
"""

from __future__ import annotations

from typing import Any

from nlql.errors import NLQLPlanError
from nlql.ir.nodes import And, Call, Compare, Expr, Literal, Not, Or, Path, Query, Ref
from nlql.plan.plan import QueryPlan, Scorer, score_key
from nlql.plan.pushdown import split_filter
from nlql.registry.core import Registry
from nlql.store.base import StoreCaps


class Planner:
    """Turns a :class:`Query` into a :class:`QueryPlan`."""

    def __init__(self, registry: Registry) -> None:
        self._registry = registry

    def plan(
        self,
        query: Query,
        *,
        granularity: str,
        caps: StoreCaps | None = None,
        field_types: dict[str, Any] | None = None,
    ) -> QueryPlan:
        caps = caps or StoreCaps()
        bindings: dict[str, Expr] = {b.name: b.expr for b in query.let}
        scorers: dict[str, Scorer] = {}
        visiting: set[str] = set()

        def visit(expr: Expr) -> None:
            if isinstance(expr, Call):
                cap = self._registry.get("function", expr.name)
                if cap is None:
                    raise NLQLPlanError(f"unknown function {expr.name!r}")
                if cap.signature is not None and not cap.signature.arity_ok(len(expr.args)):
                    raise NLQLPlanError(
                        f"{expr.name} expects {len(cap.signature.args)} args, got {len(expr.args)}"
                    )
                if cap.provides_score:
                    self._register_scorer(expr, scorers)
                for arg in expr.args:
                    visit(arg)
            elif isinstance(expr, Compare):
                visit(expr.left)
                visit(expr.right)
            elif isinstance(expr, (And, Or)):
                for operand in expr.operands:
                    visit(operand)
            elif isinstance(expr, Not):
                visit(expr.operand)
            elif isinstance(expr, Ref):
                if expr.name not in bindings:
                    raise NLQLPlanError(f"unknown alias {expr.name!r} (no matching LET binding)")
                if expr.name not in visiting:  # guard against cyclic bindings
                    visiting.add(expr.name)
                    visit(bindings[expr.name])
                    visiting.discard(expr.name)
            # Literal / Path: nothing to analyze

        for binding in query.let:
            visit(binding.expr)
        if query.where is not None:
            visit(query.where)
        for key in query.order_by:
            visit(key.expr)

        split = split_filter(query.where, caps, field_types)
        return QueryPlan(
            query=query,
            scorers=list(scorers.values()),
            bindings=bindings,
            granularity=granularity,
            pushed_filter=split.pushed,
            residual_filter=split.residual,
            store=caps.name,
        )

    @staticmethod
    def _register_scorer(call: Call, scorers: dict[str, Scorer]) -> None:
        if len(call.args) != 2:
            raise NLQLPlanError(f"{call.name}(path, \"query\") requires exactly 2 arguments")
        query_arg = call.args[1]
        if not isinstance(query_arg, Literal) or not isinstance(query_arg.value, str):
            raise NLQLPlanError(f"{call.name} query argument must be a string literal")
        # The first argument selects which *precomputed* vector to score — `content` (the
        # default vector) or `vec.<name>` (a named vector). It is NOT an evaluated expression:
        # vectors are computed at ingestion, so a transform like SIMILARITY(CUT(content), …)
        # cannot be applied at query time. Reject it loudly rather than silently ignore it.
        path_arg = call.args[0]
        if not isinstance(path_arg, Path):
            raise NLQLPlanError(
                f"{call.name}'s first argument must be `content` or `vec.<name>` (a vector "
                f"selector), not a transformed expression — vectors are precomputed at ingestion"
            )
        if path_arg.root == "vec" and path_arg.segments:
            vector_name = path_arg.segments[0]
        elif path_arg.root in ("content", "default") and not path_arg.segments:
            vector_name = "default"
        else:
            raise NLQLPlanError(
                f"{call.name}'s first argument must be `content` or `vec.<name>`, "
                f"got {path_arg.dotted!r}"
            )
        path_str = path_arg.dotted
        key = score_key(call)
        scorers.setdefault(
            key,
            Scorer(
                key=key,
                query_text=query_arg.value,
                path=path_str,
                call=call,
                vector_name=vector_name,
            ),
        )
