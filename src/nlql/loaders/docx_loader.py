"""DOCX loader (python-docx, optional)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from nlql.errors import NLQLError
from nlql.model import Document


class DocxLoader:
    """Loads a ``.docx`` file's paragraph text as a single document."""

    def load(
        self, source: str | Path, *, metadata: dict[str, Any] | None = None
    ) -> Iterable[Document]:
        try:
            import docx
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise NLQLError("DocxLoader requires python-docx: pip install python-nlql[loaders]") from e
        path = Path(source)
        document = docx.Document(str(path))
        text = "\n".join(p.text for p in document.paragraphs if p.text and p.text.strip())
        yield Document.from_text(
            text, id=path.stem, metadata={**(metadata or {}), "source": str(path)}
        )
