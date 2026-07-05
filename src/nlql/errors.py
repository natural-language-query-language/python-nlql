"""Exception hierarchy for NLQL.

All public errors derive from :class:`NLQLError` so callers can catch the whole
family with a single ``except``. Each stage of the pipeline (parse → validate →
plan → execute) raises its own specific subclass.
"""

from __future__ import annotations


class NLQLError(Exception):
    """Base class for every error raised by NLQL."""


class NLQLParseError(NLQLError):
    """Raised when an NLQL string cannot be parsed.

    Carries optional source-location info so front-ends can render a helpful
    pointer at the offending token.
    """

    def __init__(
        self,
        message: str,
        *,
        line: int | None = None,
        column: int | None = None,
        context: str | None = None,
    ) -> None:
        self.message = message
        self.line = line
        self.column = column
        self.context = context
        location = ""
        if line is not None:
            location = f" (line {line}" + (f", column {column}" if column is not None else "") + ")"
        full = f"{message}{location}"
        if context:
            full = f"{full}\n{context}"
        super().__init__(full)


class NLQLSchemaError(NLQLError):
    """Raised when a Query IR document is structurally invalid."""


class NLQLRegistryError(NLQLError):
    """Raised on invalid registration or lookup of a capability."""


class NLQLTypeError(NLQLError):
    """Raised when a comparison / function receives incompatible operand types."""


class NLQLPlanError(NLQLError):
    """Raised when a query cannot be planned against the target store."""


class NLQLExecutionError(NLQLError):
    """Raised when evaluating a planned query fails at runtime."""


class NLQLEmbeddingError(NLQLError):
    """Raised when an embedder backend fails to produce vectors."""
