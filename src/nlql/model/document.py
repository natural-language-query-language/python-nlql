"""Document — the unit of ingestion.

A document owns one or more :class:`Payload` objects (multi-modal) plus opaque
business ``metadata``. System fields (doc id, unit kind, span, scores) live on
:class:`~nlql.model.unit.Unit`, never mixed into user metadata — fixing the
reference implementation's habit of writing ``metadata["similarity"]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nlql.model.payload import Payload


@dataclass(slots=True)
class Document:
    """A source document to be ingested.

    Args:
        id: Stable, caller-provided identifier. Must be unique within a store.
        payloads: One or more content payloads (usually a single text payload).
        metadata: Opaque business metadata, addressed in queries via ``meta.*``.
        source: Optional provenance hint (file path, URL, …).
    """

    id: str
    payloads: list[Payload]
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str | None = None

    def __post_init__(self) -> None:
        if not self.payloads:
            raise ValueError("Document must have at least one payload")

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        id: str,
        metadata: dict[str, Any] | None = None,
        source: str | None = None,
    ) -> Document:
        """Build a single-payload text document."""
        return cls(id=id, payloads=[Payload.text(text)], metadata=metadata or {}, source=source)
