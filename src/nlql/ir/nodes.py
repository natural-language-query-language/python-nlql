"""Query IR — the canonical, JSON-serializable form of every NLQL query.

The NLQL string front-end, the Python Query Builder, and LLM function-calling all
compile to this one representation. Expression nodes are orthogonal and evaluate
uniformly (no special cases):

    Literal   a constant                     42, "draft", true
    Path      a field path                   content, meta.status
    Ref       a LET-bound alias              relevance
    Call      a function/operator call       SIMILARITY(content, "…"), CONTAINS(content, "x")
    Compare   an infix comparison            relevance >= 0.8
    And/Or/Not boolean composition           a AND (b OR NOT c)

A top-level :class:`Query` bundles ``select / let / where / order_by / limit``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, Union

from nlql.errors import NLQLSchemaError

# Recursive expression union.
Expr = Union["Literal", "Path", "Ref", "Call", "Compare", "And", "Or", "Not"]

_COMPARE_OPS = frozenset({"==", "!=", "<", ">", "<=", ">="})
_SELECT_UNITS = frozenset({"document", "chunk", "sentence"})
LiteralValue = str | int | float | bool | None


# --------------------------------------------------------------------------- #
# Expression nodes                                                            #
# --------------------------------------------------------------------------- #
@dataclass
class Literal:
    """A constant scalar value."""

    value: LiteralValue
    KIND: ClassVar[str] = "literal"

    def to_dict(self) -> dict[str, Any]:
        return {"node": self.KIND, "value": self.value}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Literal:
        if "value" not in d:
            raise NLQLSchemaError("literal node requires 'value'")
        return cls(value=d["value"])


@dataclass
class Path:
    """A field path: a root plus zero or more dotted segments.

    Roots in v1: ``content`` (the unit text) and ``meta`` (business metadata),
    e.g. ``Path("meta", ["status"])`` for ``meta.status``.
    """

    root: str
    segments: list[str] = field(default_factory=list)
    KIND: ClassVar[str] = "path"

    def __post_init__(self) -> None:
        if not self.root:
            raise NLQLSchemaError("path node requires a non-empty root")

    @property
    def dotted(self) -> str:
        return ".".join([self.root, *self.segments])

    def to_dict(self) -> dict[str, Any]:
        return {"node": self.KIND, "root": self.root, "segments": list(self.segments)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Path:
        return cls(root=d["root"], segments=list(d.get("segments", [])))


@dataclass
class Ref:
    """A reference to a LET-bound alias."""

    name: str
    KIND: ClassVar[str] = "ref"

    def to_dict(self) -> dict[str, Any]:
        return {"node": self.KIND, "name": self.name}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Ref:
        return cls(name=d["name"])


@dataclass
class Call:
    """A function or operator call, resolved against the registry by name."""

    name: str
    args: list[Expr] = field(default_factory=list)
    KIND: ClassVar[str] = "call"

    def to_dict(self) -> dict[str, Any]:
        return {"node": self.KIND, "name": self.name, "args": [a.to_dict() for a in self.args]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Call:
        return cls(name=d["name"], args=[expr_from_dict(a) for a in d.get("args", [])])


@dataclass
class Compare:
    """An infix comparison; ``op`` in ``== != < > <= >=``."""

    op: str
    left: Expr
    right: Expr
    KIND: ClassVar[str] = "compare"

    def __post_init__(self) -> None:
        if self.op not in _COMPARE_OPS:
            raise NLQLSchemaError(f"unknown comparison op {self.op!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.KIND,
            "op": self.op,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Compare:
        return cls(op=d["op"], left=expr_from_dict(d["left"]), right=expr_from_dict(d["right"]))


@dataclass
class And:
    """Boolean AND over two or more operands."""

    operands: list[Expr]
    KIND: ClassVar[str] = "and"

    def to_dict(self) -> dict[str, Any]:
        return {"node": self.KIND, "operands": [o.to_dict() for o in self.operands]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> And:
        return cls(operands=[expr_from_dict(o) for o in d["operands"]])


@dataclass
class Or:
    """Boolean OR over two or more operands."""

    operands: list[Expr]
    KIND: ClassVar[str] = "or"

    def to_dict(self) -> dict[str, Any]:
        return {"node": self.KIND, "operands": [o.to_dict() for o in self.operands]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Or:
        return cls(operands=[expr_from_dict(o) for o in d["operands"]])


@dataclass
class Not:
    """Boolean negation."""

    operand: Expr
    KIND: ClassVar[str] = "not"

    def to_dict(self) -> dict[str, Any]:
        return {"node": self.KIND, "operand": self.operand.to_dict()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Not:
        return cls(operand=expr_from_dict(d["operand"]))


_EXPR_DISPATCH: dict[str, Callable[[dict[str, Any]], Expr]] = {
    Literal.KIND: Literal.from_dict,
    Path.KIND: Path.from_dict,
    Ref.KIND: Ref.from_dict,
    Call.KIND: Call.from_dict,
    Compare.KIND: Compare.from_dict,
    And.KIND: And.from_dict,
    Or.KIND: Or.from_dict,
    Not.KIND: Not.from_dict,
}


def expr_from_dict(d: Any) -> Expr:
    """Rebuild an expression node from its dict form."""
    if not isinstance(d, dict) or "node" not in d:
        raise NLQLSchemaError(f"expression must be an object with a 'node' tag, got {d!r}")
    ctor = _EXPR_DISPATCH.get(d["node"])
    if ctor is None:
        raise NLQLSchemaError(f"unknown expression node {d['node']!r}")
    return ctor(d)


# --------------------------------------------------------------------------- #
# Query structure                                                            #
# --------------------------------------------------------------------------- #
@dataclass
class Select:
    """The SELECT clause: base granularity plus optional SPAN window.

    ``window`` is the ``SPAN(<unit>, window => N)`` context radius; ``None`` means
    return the base unit itself.
    """

    unit: str
    window: int | None = None

    def __post_init__(self) -> None:
        if self.unit not in _SELECT_UNITS:
            raise NLQLSchemaError(
                f"unknown SELECT unit {self.unit!r}; expected one of {sorted(_SELECT_UNITS)}"
            )
        if self.window is not None and self.window < 0:
            raise NLQLSchemaError("SPAN window must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"unit": self.unit}
        if self.window is not None:
            d["window"] = self.window
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Select:
        return cls(unit=d["unit"], window=d.get("window"))


@dataclass
class Binding:
    """A ``LET name = expr`` named binding."""

    name: str
    expr: Expr

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "expr": self.expr.to_dict()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Binding:
        return cls(name=d["name"], expr=expr_from_dict(d["expr"]))


@dataclass
class OrderKey:
    """One ``ORDER BY`` key."""

    expr: Expr
    desc: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"expr": self.expr.to_dict(), "desc": self.desc}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> OrderKey:
        return cls(expr=expr_from_dict(d["expr"]), desc=bool(d.get("desc", False)))


@dataclass
class Query:
    """The canonical Query IR."""

    select: Select
    let: list[Binding] = field(default_factory=list)
    where: Expr | None = None
    order_by: list[OrderKey] = field(default_factory=list)
    limit: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"select": self.select.to_dict()}
        if self.let:
            d["let"] = [b.to_dict() for b in self.let]
        if self.where is not None:
            d["where"] = self.where.to_dict()
        if self.order_by:
            d["order_by"] = [k.to_dict() for k in self.order_by]
        if self.limit is not None:
            d["limit"] = self.limit
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Query:
        if "select" not in d:
            raise NLQLSchemaError("query requires a 'select' clause")
        where = d.get("where")
        return cls(
            select=Select.from_dict(d["select"]),
            let=[Binding.from_dict(b) for b in d.get("let", [])],
            where=expr_from_dict(where) if where is not None else None,
            order_by=[OrderKey.from_dict(k) for k in d.get("order_by", [])],
            limit=d.get("limit"),
        )

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls, s: str) -> Query:
        return cls.from_dict(json.loads(s))
