"""Transform a Lark parse tree into Query IR.

Identifier references are emitted uniformly as :class:`~nlql.ir.nodes.Path` nodes;
a later resolution pass (in :mod:`nlql.lang.parser`) rewrites bare paths that match a
LET alias into :class:`~nlql.ir.nodes.Ref` nodes. This keeps the transformer local and
context-free — it does not need to know the alias set.
"""

from __future__ import annotations

from typing import Any

from lark import Token, Transformer

from nlql.errors import NLQLParseError
from nlql.ir.nodes import (
    And,
    Binding,
    Call,
    Compare,
    Literal,
    Not,
    Or,
    OrderKey,
    Path,
    Query,
    Select,
)


class NLQLTransformer(Transformer):
    """Bottom-up transformer producing an (unresolved) :class:`Query`."""

    # -- literals --------------------------------------------------------------
    def str_lit(self, items: list[Token]) -> Literal:
        raw = str(items[0])
        return Literal(raw[1:-1])  # strip surrounding quotes

    def num_lit(self, items: list[Token]) -> Literal:
        raw = str(items[0])
        return Literal(float(raw) if "." in raw else int(raw))

    def bool_lit(self, items: list[Token]) -> Literal:
        return Literal(str(items[0]) == "true")

    def null_lit(self, _items: list[Token]) -> Literal:
        return Literal(None)

    # -- references & calls ----------------------------------------------------
    def path(self, items: list[Token]) -> Path:
        names = [str(t) for t in items]
        return Path(root=names[0], segments=names[1:])

    def arglist(self, items: list[Any]) -> list[Any]:
        return list(items)

    def call(self, items: list[Any]) -> Call:
        name = str(items[0])
        args = items[1] if len(items) > 1 else []
        return Call(name=name, args=args)

    # -- predicates & comparisons ---------------------------------------------
    def infix_pred(self, items: list[Any]) -> Call:
        left, op, right = items[0], str(items[1]), items[2]
        return Call(name=op.upper(), args=[left, right])

    def compare(self, items: list[Any]) -> Compare:
        left, op, right = items[0], str(items[1]), items[2]
        return Compare(op=op, left=left, right=right)

    # -- boolean composition (flatten same-op chains) --------------------------
    def and_op(self, items: list[Any]) -> And:
        left, right = items
        if isinstance(left, And):
            left.operands.append(right)
            return left
        return And([left, right])

    def or_op(self, items: list[Any]) -> Or:
        left, right = items
        if isinstance(left, Or):
            left.operands.append(right)
            return left
        return Or([left, right])

    def not_op(self, items: list[Any]) -> Not:
        return Not(items[0])

    # -- SELECT ----------------------------------------------------------------
    def plain_unit(self, items: list[Token]) -> Select:
        return Select(unit=str(items[0]).lower())

    def named_window(self, items: list[Token]) -> int:
        keyword, number = str(items[0]), str(items[1])
        if keyword != "window":
            raise NLQLParseError(f"expected 'window =>' in SPAN, got {keyword!r} =>")
        return int(number)

    def pos_window(self, items: list[Token]) -> int:
        return int(str(items[0]))

    def span_unit(self, items: list[Any]) -> Select:
        unit = str(items[0]).lower()
        window = items[1] if len(items) > 1 else 1  # SPAN(X) defaults to window 1
        return Select(unit=unit, window=window)

    # -- clauses ---------------------------------------------------------------
    def binding(self, items: list[Any]) -> Binding:
        return Binding(name=str(items[0]), expr=items[1])

    def let_clause(self, items: list[Binding]) -> tuple[str, list[Binding]]:
        return ("let", list(items))

    def where_clause(self, items: list[Any]) -> tuple[str, Any]:
        return ("where", items[0])

    def order_key(self, items: list[Any]) -> OrderKey:
        expr = items[0]
        desc = len(items) > 1 and str(items[1]) == "DESC"
        return OrderKey(expr=expr, desc=desc)

    def order_clause(self, items: list[OrderKey]) -> tuple[str, list[OrderKey]]:
        return ("order", list(items))

    def limit_clause(self, items: list[Token]) -> tuple[str, int]:
        return ("limit", int(str(items[0])))

    # -- top level -------------------------------------------------------------
    def query(self, items: list[Any]) -> Query:
        select: Select = items[0]
        q = Query(select=select)
        for tag, val in items[1:]:
            if tag == "let":
                q.let = val
            elif tag == "where":
                q.where = val
            elif tag == "order":
                q.order_by = val
            elif tag == "limit":
                q.limit = val
        return q
