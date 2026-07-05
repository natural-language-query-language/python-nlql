"""Vectorized metadata column filtering for LocalStore (M6 Python-level optimization).

Applies a pushed metadata filter as a numpy boolean mask instead of a per-row Python loop.
Value coercion — the profiled hotspot — runs once per *distinct* field value (not per row),
and masks compose with numpy ``&`` / ``|`` / ``~``. Semantics are identical to
``matches_filter``: both go through :func:`~nlql.types.coerce.compare_values`, so results
match the per-row path and every other store exactly.

Columns are built lazily per referenced field and cached until the next upsert.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nlql.ir.nodes import And, Compare, Expr, Not, Or
from nlql.plan.pushdown import normalized_compare
from nlql.types.coerce import compare_values


class MetadataColumns:
    """Lazily-built, cached metadata columns aligned with a store's id order."""

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._units: dict[str, Any] = {}
        self._cache: dict[str, np.ndarray] = {}

    def reset(self, ids: list[str], units: dict[str, Any]) -> None:
        """Point at a new id order / unit set and drop cached columns."""
        self._ids = ids
        self._units = units
        self._cache = {}

    def _column(self, field: str) -> np.ndarray:
        col = self._cache.get(field)
        if col is None:
            segments = field.split(".")
            values: list[Any] = []
            for uid in self._ids:
                current: Any = self._units[uid].metadata
                for seg in segments:
                    current = current.get(seg) if isinstance(current, dict) else None
                values.append(current)
            col = np.array(values, dtype=object)
            self._cache[field] = col
        return col

    def mask(self, expr: Expr) -> np.ndarray:
        """A boolean mask over the current id order for a pushed metadata filter."""
        if isinstance(expr, Compare):
            field, op, value = normalized_compare(expr)
            return self._compare_mask(self._column(field), op, value)
        if isinstance(expr, And):
            result = self.mask(expr.operands[0])
            for operand in expr.operands[1:]:
                result = result & self.mask(operand)
            return result
        if isinstance(expr, Or):
            result = self.mask(expr.operands[0])
            for operand in expr.operands[1:]:
                result = result | self.mask(operand)
            return result
        if isinstance(expr, Not):
            return ~self.mask(expr.operand)
        raise ValueError(f"cannot build a column mask for {type(expr).__name__}")

    @staticmethod
    def _compare_mask(col: np.ndarray, op: str, value: Any) -> np.ndarray:
        n = len(col)
        if n == 0:
            return np.zeros(0, dtype=bool)
        try:
            # Factorize distinct values, then run compare_values once per distinct value.
            codes = np.empty(n, dtype=np.intp)
            uniques: dict[Any, int] = {}
            for i, v in enumerate(col):
                code = uniques.get(v)
                if code is None:
                    code = len(uniques)
                    uniques[v] = code
                codes[i] = code
            results = np.fromiter(
                (compare_values(op, uv, value) for uv in uniques),
                dtype=bool,
                count=len(uniques),
            )
            return results[codes]
        except TypeError:
            # Unhashable metadata value (e.g. a list) — compare per row (still correct).
            return np.array([compare_values(op, v, value) for v in col], dtype=bool)
