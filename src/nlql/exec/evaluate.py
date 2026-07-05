"""The expression evaluator — one uniform path, no special cases.

Every node evaluates to a value. The *only* dispatch that distinguishes a semantic score
from an ordinary function is a check on the capability's ``provides_score`` flag — not a
hardcoded name like the reference implementation's ``if node.operator == "SIMILAR_TO"``.
Comparisons coerce operands first (numbers/dates), and null operands drop out of ordered
comparisons (SQL-like), so a missing metadata field never crashes a query.
"""

from __future__ import annotations

from typing import Any

from nlql.errors import NLQLExecutionError
from nlql.ir.nodes import And, Call, Compare, Expr, Literal, Not, Or, Path, Ref
from nlql.model import Unit
from nlql.plan.plan import score_key
from nlql.registry.core import Registry
from nlql.types.coerce import compare_values
from nlql.types.core import TypeTag


class Evaluator:
    """Evaluates IR expressions against a single unit and a set of LET bindings."""

    def __init__(self, registry: Registry, field_types: dict[str, TypeTag] | None = None, type_handlers: dict | None = None) -> None:
        self._registry = registry
        self._field_types = field_types or {}
        self._type_handlers = type_handlers if type_handlers is not None else {}

    def eval(self, expr: Expr, unit: Unit, bindings: dict[str, Expr]) -> Any:
        if isinstance(expr, Literal):
            return expr.value
        if isinstance(expr, Path):
            return self._eval_path(expr, unit)
        if isinstance(expr, Ref):
            if expr.name not in bindings:
                raise NLQLExecutionError(f"unknown alias {expr.name!r}")
            return self.eval(bindings[expr.name], unit, bindings)
        if isinstance(expr, Call):
            return self._eval_call(expr, unit, bindings)
        if isinstance(expr, Compare):
            return self._eval_compare(expr, unit, bindings)
        if isinstance(expr, And):
            return all(self.truthy(self.eval(o, unit, bindings)) for o in expr.operands)
        if isinstance(expr, Or):
            return any(self.truthy(self.eval(o, unit, bindings)) for o in expr.operands)
        if isinstance(expr, Not):
            return not self.truthy(self.eval(expr.operand, unit, bindings))
        raise NLQLExecutionError(f"cannot evaluate node {type(expr).__name__}")

    # -- leaves ----------------------------------------------------------------
    @staticmethod
    def _eval_path(path: Path, unit: Unit) -> Any:
        if path.root == "content":
            return unit.content if not path.segments else None
        # "meta.x" and bare "x" both address business metadata.
        segments = path.segments if path.root == "meta" else [path.root, *path.segments]
        current: Any = unit.metadata
        for seg in segments:
            if isinstance(current, dict):
                current = current.get(seg)
            else:
                return None
        return current

    def _eval_call(self, call: Call, unit: Unit, bindings: dict[str, Expr]) -> Any:
        cap = self._registry.get("function", call.name)
        if cap is None:
            raise NLQLExecutionError(f"unknown function {call.name!r}")
        if cap.provides_score:
            key = score_key(call)
            if key not in unit.scores:
                raise NLQLExecutionError(
                    f"score for {call.name} was not computed (planner/executor mismatch)"
                )
            return unit.scores[key]
        args = [self.eval(a, unit, bindings) for a in call.args]
        return cap.impl(*args)

    # -- comparison ------------------------------------------------------------
    def _eval_compare(self, node: Compare, unit: Unit, bindings: dict[str, Expr]) -> bool:
        left = self.eval(node.left, unit, bindings)
        right = self.eval(node.right, unit, bindings)
        hint = self._hint_for(node.left, node.right)
        return compare_values(node.op, left, right, hint, type_handlers=self._type_handlers)

    def _field_hint(self, node: Expr) -> TypeTag | None:
        """The declared type of a metadata path, if any (drives typed comparison)."""
        if isinstance(node, Path) and node.root != "content":
            segments = node.segments if node.root == "meta" else [node.root, *node.segments]
            return self._field_types.get(".".join(segments))
        return None

    def _hint_for(self, left_node: Expr, right_node: Expr) -> Any:
        """Resolve comparison hint: field_types declaration first, then explicit
        Literal type_hint (e.g. ``DATE '2024-01-01'`` → hint='date')."""
        hint = self._field_hint(left_node) or self._field_hint(right_node)
        if hint:
            return hint
        if isinstance(right_node, Literal) and right_node.type_hint:
            return right_node.type_hint
        if isinstance(left_node, Literal) and left_node.type_hint:
            return left_node.type_hint
        return None

    @staticmethod
    def truthy(value: Any) -> bool:
        return bool(value) if value is not None else False
