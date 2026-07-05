"""Built-in functions seeded into the global registry.

These are pure, stateless functions (predicates + scalars). ``SIMILARITY`` is special:
it is a *provider-backed* function — it has no row-wise ``impl``; its value is computed
by the recall stage and read from ``Unit.scores``. Registering it here (rather than
hardcoding it in the evaluator) is what keeps the evaluator free of special cases.
"""

from __future__ import annotations

import re

from nlql.registry.core import GLOBAL_REGISTRY
from nlql.types.core import Signature
from nlql.types.core import TypeTag as T


# --------------------------------------------------------------------------- #
# Predicates — return BOOL. Exposed with infix sugar in the grammar.          #
# --------------------------------------------------------------------------- #
@GLOBAL_REGISTRY.function(
    "CONTAINS", signature=Signature((T.TEXT, T.TEXT), T.BOOL), pushdownable=True
)
def _contains(text: object, needle: object) -> bool:
    """Case-insensitive substring test."""
    return str(needle).lower() in str(text).lower()


@GLOBAL_REGISTRY.function("MATCH", signature=Signature((T.TEXT, T.TEXT), T.BOOL))
def _match(text: object, pattern: object) -> bool:
    """Regular-expression search (``re.search`` semantics)."""
    return re.search(str(pattern), str(text)) is not None


@GLOBAL_REGISTRY.function("LIKE", signature=Signature((T.TEXT, T.TEXT), T.BOOL))
def _like(text: object, pattern: object) -> bool:
    """SQL ``LIKE`` matching: ``%`` = any run, ``_`` = any single char (anchored)."""
    escaped = re.escape(str(pattern)).replace("%", ".*").replace("_", ".")
    return re.match(f"^{escaped}$", str(text), flags=re.DOTALL) is not None


# --------------------------------------------------------------------------- #
# Scalars — return NUMBER / TEXT.                                             #
# --------------------------------------------------------------------------- #
@GLOBAL_REGISTRY.function("LENGTH", signature=Signature((T.TEXT,), T.NUMBER))
def _length(text: object) -> int:
    """Character length of the argument."""
    return len(str(text))


@GLOBAL_REGISTRY.function("COUNT", signature=Signature((T.TEXT, T.TEXT), T.NUMBER))
def _count(text: object, needle: object) -> int:
    """Case-insensitive count of non-overlapping occurrences of ``needle``."""
    hay, sub = str(text).lower(), str(needle).lower()
    return hay.count(sub) if sub else 0


@GLOBAL_REGISTRY.function("LOWER", signature=Signature((T.TEXT,), T.TEXT))
def _lower(text: object) -> str:
    """Lowercase the argument."""
    return str(text).lower()


@GLOBAL_REGISTRY.function("UPPER", signature=Signature((T.TEXT,), T.TEXT))
def _upper(text: object) -> str:
    """Uppercase the argument."""
    return str(text).upper()


# --------------------------------------------------------------------------- #
# Provider-backed scoring function — value comes from the recall stage.       #
# --------------------------------------------------------------------------- #
GLOBAL_REGISTRY.register(
    "function",
    "SIMILARITY",
    impl=None,
    signature=Signature((T.ANY, T.TEXT), T.NUMBER),
    provides_score=True,
    pushdownable=True,
    doc="Semantic similarity in [-1, 1] (raw cosine). Provided by the index recall stage.",
)
