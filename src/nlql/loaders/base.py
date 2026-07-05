"""Document loaders — turn files (txt / md / docx / pdf / …) into :class:`Document`.

A loader reads a source and yields documents; the ingestion pipeline then splits and embeds
them as usual. Dispatch is by file extension against a registry, so adding a format is
``register_loader(MyLoader(), ".epub")`` — or pass ``loader=`` explicitly. Format-specific
dependencies (``python-docx``, ``pypdf``) are imported lazily inside each loader, so importing
``nlql.loaders`` never requires them.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from nlql.errors import NLQLError
from nlql.model import Document


@runtime_checkable
class Loader(Protocol):
    """Reads a source path into one or more documents."""

    def load(self, source: str | Path, *, metadata: dict[str, Any] | None = None) -> Iterable[Document]:
        ...


_LOADERS: dict[str, Loader] = {}


def register_loader(loader: Loader, *extensions: str) -> None:
    """Register ``loader`` for one or more file extensions (e.g. ``".pdf"``)."""
    for ext in extensions:
        _LOADERS[ext.lower()] = loader


def loader_for(extension: str) -> Loader | None:
    return _LOADERS.get(extension.lower())


def load_documents(
    source: str | Path,
    *,
    loader: Loader | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[Document]:
    """Load a file into documents, picking a loader by extension unless one is given."""
    path = Path(source)
    chosen = loader if loader is not None else _LOADERS.get(path.suffix.lower())
    if chosen is None:
        raise NLQLError(
            f"no loader registered for {path.suffix!r}; pass loader= or register_loader(...)"
        )
    return list(chosen.load(path, metadata=metadata))
