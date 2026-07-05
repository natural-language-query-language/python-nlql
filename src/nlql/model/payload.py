"""Modality and Payload — the modality-agnostic content container.

A :class:`Payload` replaces the reference implementation's bare ``content: str``.
Text is simply one modality; image / blob carriers are first-class from day one,
even though only text is *embedded* in v1.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Modality(StrEnum):
    """The kind of content a payload carries.

    A ``StrEnum`` so it compares and serializes as a plain string.
    """

    TEXT = "text"
    IMAGE = "image"
    BLOB = "blob"


@dataclass(slots=True)
class Payload:
    """A single piece of content plus its modality.

    Args:
        modality: What kind of data ``data`` holds.
        data: ``str`` for text; ``bytes`` or a URI ``str`` for image / blob.
        mime: Optional MIME type (e.g. ``"image/png"``), useful for adapters.
    """

    modality: Modality
    data: str | bytes
    mime: str | None = None

    def __post_init__(self) -> None:
        if self.modality is Modality.TEXT and not isinstance(self.data, str):
            raise TypeError("TEXT payload data must be a str")

    @classmethod
    def text(cls, s: str) -> Payload:
        """Convenience constructor for a text payload."""
        return cls(modality=Modality.TEXT, data=s)

    @property
    def is_text(self) -> bool:
        return self.modality is Modality.TEXT

    @property
    def as_text(self) -> str:
        """Return textual data, or ``""`` for non-text payloads."""
        return self.data if self.modality is Modality.TEXT and isinstance(self.data, str) else ""
