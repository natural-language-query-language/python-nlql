"""Plain-text / Markdown loader."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from nlql.model import Document


class TextLoader:
    """Loads a UTF-8 text file as a single document."""

    def load(
        self, source: str | Path, *, metadata: dict[str, Any] | None = None
    ) -> Iterable[Document]:
        path = Path(source)
        text = path.read_text(encoding="utf-8")
        yield Document.from_text(
            text, id=path.stem, metadata={**(metadata or {}), "source": str(path)}
        )
