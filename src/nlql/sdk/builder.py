"""A small, type-safe Query Builder that compiles to the same Query IR.

Mirrors the NLQL string surface with Python operators, e.g.::

    select("sentence")
        .let("relevance", similarity("content", "AI agents"))
        .where(F("relevance") >= 0.8, Meta("status") != "draft")
        .order_by("relevance", desc=True)
        .limit(5)
        .build()
"""

from __future__ import annotations

from typing import Any

from nlql.ir.nodes import (
    And,
    Binding,
    Call,
    Compare,
    Expr,
    Literal,
    Not,
    Or,
    OrderKey,
    Path,
    Query,
    Ref,
    Select,
)


def _path_from_dotted(dotted: str) -> Path:
    parts = dotted.split(".")
    if parts[0] == "meta":
        return Path("meta", parts[1:])
    return Path(parts[0], parts[1:])


def _lift(value: Any) -> Expr:
    """Coerce a Python value / E / Expr into an IR expression."""
    if isinstance(value, E):
        return value.expr
    if isinstance(value, (Literal, Path, Ref, Call, Compare, And, Or, Not)):
        return value
    return Literal(value)


class E:
    """A fluent wrapper over an IR expression, overloading Python operators."""

    __hash__ = None  # type: ignore[assignment]  # comparison ops build IR, not booleans

    def __init__(self, expr: Expr) -> None:
        self.expr = expr

    # comparisons -> Compare
    def __ge__(self, other: Any) -> E:
        return E(Compare(">=", self.expr, _lift(other)))

    def __gt__(self, other: Any) -> E:
        return E(Compare(">", self.expr, _lift(other)))

    def __le__(self, other: Any) -> E:
        return E(Compare("<=", self.expr, _lift(other)))

    def __lt__(self, other: Any) -> E:
        return E(Compare("<", self.expr, _lift(other)))

    def __eq__(self, other: Any) -> E:  # type: ignore[override]
        return E(Compare("==", self.expr, _lift(other)))

    def __ne__(self, other: Any) -> E:  # type: ignore[override]
        return E(Compare("!=", self.expr, _lift(other)))

    # boolean composition -> And / Or / Not
    def __and__(self, other: Any) -> E:
        rhs = _lift(other)
        if isinstance(self.expr, And):
            return E(And([*self.expr.operands, rhs]))
        return E(And([self.expr, rhs]))

    def __or__(self, other: Any) -> E:
        rhs = _lift(other)
        if isinstance(self.expr, Or):
            return E(Or([*self.expr.operands, rhs]))
        return E(Or([self.expr, rhs]))

    def __invert__(self) -> E:
        return E(Not(self.expr))

    # predicate helpers
    def contains(self, text: Any) -> E:
        return E(Call("CONTAINS", [self.expr, _lift(text)]))

    def matches(self, pattern: Any) -> E:
        return E(Call("MATCH", [self.expr, _lift(pattern)]))

    def like(self, pattern: Any) -> E:
        return E(Call("LIKE", [self.expr, _lift(pattern)]))


# -- expression constructors --------------------------------------------------
content = E(Path("content"))


def field(dotted: str) -> E:
    return E(_path_from_dotted(dotted))


def Meta(key: str) -> E:  # noqa: N802 - deliberate DSL capitalization
    return E(Path("meta", [key]))


def F(name: str) -> E:  # noqa: N802
    """Reference a LET-bound alias."""
    return E(Ref(name))


def similarity(path: str | E, query: str) -> E:
    target = _lift(path) if isinstance(path, E) else _path_from_dotted(path)
    return E(Call("SIMILARITY", [target, Literal(query)]))


def contains(path: str | E, text: str) -> E:
    target = _lift(path) if isinstance(path, E) else _path_from_dotted(path)
    return E(Call("CONTAINS", [target, Literal(text)]))


def length(path: str | E) -> E:
    target = _lift(path) if isinstance(path, E) else _path_from_dotted(path)
    return E(Call("LENGTH", [target]))


# -- the builder --------------------------------------------------------------
class QueryBuilder:
    """Accumulates clauses and compiles to a :class:`~nlql.ir.nodes.Query`."""

    def __init__(self, unit: str, window: int | None = None) -> None:
        self._select = Select(unit, window)
        self._let: list[Binding] = []
        self._where: list[Expr] = []
        self._order: list[OrderKey] = []
        self._limit: int | None = None

    def let(self, name: str, expr: Any) -> QueryBuilder:
        self._let.append(Binding(name, _lift(expr)))
        return self

    def where(self, *conditions: Any) -> QueryBuilder:
        self._where.extend(_lift(c) for c in conditions)
        return self

    def order_by(self, key: Any, *, desc: bool = False) -> QueryBuilder:
        expr = Ref(key) if isinstance(key, str) else _lift(key)
        self._order.append(OrderKey(expr, desc=desc))
        return self

    def limit(self, n: int) -> QueryBuilder:
        self._limit = n
        return self

    def build(self) -> Query:
        where: Expr | None
        if not self._where:
            where = None
        elif len(self._where) == 1:
            where = self._where[0]
        else:
            where = And(list(self._where))
        return Query(
            select=self._select,
            let=self._let,
            where=where,
            order_by=self._order,
            limit=self._limit,
        )


def select(unit: str, window: int | None = None) -> QueryBuilder:
    """Start building a query at the given granularity."""
    return QueryBuilder(unit, window)
