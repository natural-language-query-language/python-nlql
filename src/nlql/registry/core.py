"""The unified capability registry.

A single :class:`Registry` type holds every extension point — functions, splitters,
embedders, modalities — under one registration protocol. Built-in capabilities and
user extensions travel the exact same path (no placeholder dead code like the
reference implementation's ``_placeholder_similar_to``).

Scope / precedence is modelled by a **parent chain** rather than a ``scope`` string:
the process-wide :data:`GLOBAL_REGISTRY` is the root; an engine instance holds a
:meth:`Registry.child`, so instance registrations naturally shadow global ones.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from nlql.errors import NLQLRegistryError
from nlql.types.core import Signature

# Capability kinds. Predicates (CONTAINS/MATCH/LIKE) are ordinary *functions* returning
# BOOL, not a separate kind — infix syntax is mere parser sugar over a function call.
CAPABILITY_KINDS = frozenset({"function", "splitter", "embedder", "modality"})


@dataclass(frozen=True, slots=True)
class Capability:
    """A single registered extension point.

    Args:
        kind: One of :data:`CAPABILITY_KINDS`.
        name: Display name (lookup is case-insensitive).
        impl: The implementation object (callable, splitter fn, embedder, …). May be
            ``None`` for provider-backed functions whose value comes from another stage.
        signature: Optional declared types, used for arity/type checks and pushdown.
        provides_score: When ``True`` this is a function whose value is supplied by the
            recall stage and read from ``Unit.scores`` (e.g. ``SIMILARITY``) — it is not
            invoked row-by-row.
        pushdownable: Hint to the Planner that this capability can be translated to an
            external store's native query (used in M2).
        doc: Human-readable description.
    """

    kind: str
    name: str
    impl: Any = None
    signature: Signature | None = None
    provides_score: bool = False
    pushdownable: bool = False
    doc: str | None = None


class Registry:
    """A capability registry, optionally chained to a parent for scoped overrides."""

    def __init__(self, parent: Registry | None = None) -> None:
        self._parent = parent
        self._caps: dict[tuple[str, str], Capability] = {}

    @staticmethod
    def _key(kind: str, name: str) -> tuple[str, str]:
        return (kind, name.upper())

    def register(
        self,
        kind: str,
        name: str,
        impl: Any = None,
        *,
        signature: Signature | None = None,
        provides_score: bool = False,
        pushdownable: bool = False,
        doc: str | None = None,
        overwrite: bool = False,
    ) -> Capability:
        """Register a capability into *this* registry.

        Raises:
            NLQLRegistryError: on unknown ``kind``, empty ``name``, or a same-registry
                name clash without ``overwrite=True``. (Shadowing a *parent* entry is
                never a clash — that is how instance overrides work.)
        """
        if kind not in CAPABILITY_KINDS:
            raise NLQLRegistryError(
                f"Unknown capability kind {kind!r}; expected one of {sorted(CAPABILITY_KINDS)}"
            )
        if not name or not name.strip():
            raise NLQLRegistryError("Capability name must be a non-empty string")
        key = self._key(kind, name)
        if key in self._caps and not overwrite:
            raise NLQLRegistryError(
                f"{kind} {name!r} already registered in this registry; pass overwrite=True to replace"
            )
        cap = Capability(
            kind=kind,
            name=name,
            impl=impl,
            signature=signature,
            provides_score=provides_score,
            pushdownable=pushdownable,
            doc=doc,
        )
        self._caps[key] = cap
        return cap

    def get(self, kind: str, name: str) -> Capability | None:
        """Resolve a capability, checking this registry then its parent chain."""
        cap = self._caps.get(self._key(kind, name))
        if cap is not None:
            return cap
        if self._parent is not None:
            return self._parent.get(kind, name)
        return None

    def has(self, kind: str, name: str) -> bool:
        return self.get(kind, name) is not None

    def names(self, kind: str) -> list[str]:
        """All capability names of ``kind`` visible here (child shadows parent)."""
        merged: dict[str, str] = {}
        if self._parent is not None:
            for n in self._parent.names(kind):
                merged[n.upper()] = n
        for (k, up), cap in self._caps.items():
            if k == kind:
                merged[up] = cap.name
        return sorted(merged.values())

    def child(self) -> Registry:
        """Create an instance-scoped registry that overrides this one."""
        return Registry(parent=self)

    # ------------------------------------------------------------------ #
    # Ergonomic decorators — the user-facing sugar over ``register``.    #
    # ------------------------------------------------------------------ #

    def function(
        self,
        name: str,
        *,
        signature: Signature | None = None,
        provides_score: bool = False,
        pushdownable: bool = False,
        doc: str | None = None,
        overwrite: bool = False,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator registering a scalar/predicate function."""

        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.register(
                "function",
                name,
                fn,
                signature=signature,
                provides_score=provides_score,
                pushdownable=pushdownable,
                doc=doc or fn.__doc__,
                overwrite=overwrite,
            )
            return fn

        return deco

    def splitter(
        self, name: str, *, overwrite: bool = False
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator registering a text splitter (``str -> list[str]``)."""

        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.register("splitter", name, fn, doc=fn.__doc__, overwrite=overwrite)
            return fn

        return deco

    def embedder(
        self, name: str, *, overwrite: bool = False
    ) -> Callable[[Any], Any]:
        """Decorator/registrar for an embedder instance or class."""

        def deco(obj: Any) -> Any:
            self.register("embedder", name, obj, overwrite=overwrite)
            return obj

        return deco


# Process-wide root registry. Built-in functions seed themselves here on import.
GLOBAL_REGISTRY = Registry()


def register_function(
    name: str,
    *,
    signature: Signature | None = None,
    provides_score: bool = False,
    pushdownable: bool = False,
    doc: str | None = None,
    overwrite: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a function into the process-wide :data:`GLOBAL_REGISTRY`."""
    return GLOBAL_REGISTRY.function(
        name,
        signature=signature,
        provides_score=provides_score,
        pushdownable=pushdownable,
        doc=doc,
        overwrite=overwrite,
    )


def register_splitter(
    name: str, *, overwrite: bool = False
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a splitter into the process-wide :data:`GLOBAL_REGISTRY`."""
    return GLOBAL_REGISTRY.splitter(name, overwrite=overwrite)
