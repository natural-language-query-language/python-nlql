"""PDF loader (pypdf, optional).

Extracts page text, and optionally the embedded images. ``by_page=True`` yields one document
per page (with a ``page`` field) for page-level retrieval; the default concatenates all pages.
``extract_images=True`` yields a single document whose payloads are the text **plus** every
embedded image — the existing pipeline then indexes the text as sentence units and each image
as an image unit (needs a multimodal embedder).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from nlql.errors import NLQLError
from nlql.model import Document, Modality, Payload


class PdfLoader:
    """Loads a ``.pdf`` file's extracted text (and optionally its images)."""

    def __init__(self, *, by_page: bool = False, extract_images: bool = False) -> None:
        self._by_page = by_page
        self._extract_images = extract_images

    def load(
        self, source: str | Path, *, metadata: dict[str, Any] | None = None
    ) -> Iterable[Document]:
        try:
            from pypdf import PdfReader
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise NLQLError("PdfLoader requires pypdf: pip install python-nlql[loaders]") from e
        path = Path(source)
        reader = PdfReader(str(path))
        base = {**(metadata or {}), "source": str(path)}

        if self._extract_images:
            yield self._with_images(reader, path, base)
        elif self._by_page:
            for i, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    yield Document.from_text(text, id=f"{path.stem}#p{i}", metadata={**base, "page": i})
        else:
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
            yield Document.from_text(text, id=path.stem, metadata={**base, "pages": len(reader.pages)})

    @staticmethod
    def _with_images(reader: Any, path: Path, base: dict[str, Any]) -> Document:
        texts: list[str] = []
        payloads: list[Payload] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                texts.append(text)
            for image in page.images:
                payloads.append(Payload(modality=Modality.IMAGE, data=image.data))
        if texts:
            payloads.insert(0, Payload.text("\n".join(texts)))
        if not payloads:
            payloads = [Payload.text("")]
        n_images = sum(1 for p in payloads if p.modality is Modality.IMAGE)
        return Document(id=path.stem, payloads=payloads, metadata={**base, "images": n_images})
