"""The ingestion pipeline: normalize -> split -> embed(cache) -> Unit.

Produces indexed :class:`~nlql.model.unit.Unit` objects from documents. Splitting happens
here, at write time — the same segmentation later serves SENTENCE/SPAN queries, so the
query path never re-splits. Embedding goes through whatever embedder is supplied (normally
a :class:`~nlql.embed.cache.CachedEmbedder`), so repeated text is embedded at most once.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import cast

from nlql.embed.base import Embedder
from nlql.embed.multimodal import supports_images
from nlql.errors import NLQLRegistryError
from nlql.ingest.normalize import DefaultNormalizer, Normalizer
from nlql.model import Document, Modality, Payload, Unit
from nlql.registry.core import GLOBAL_REGISTRY, Registry


class IngestionPipeline:
    """Turns documents into embedded units at a configured base granularity."""

    def __init__(
        self,
        embedder: Embedder,
        *,
        registry: Registry = GLOBAL_REGISTRY,
        normalizer: Normalizer | None = None,
        granularity: str = "sentence",
    ) -> None:
        self._embedder = embedder
        self._registry = registry
        self._normalizer = normalizer or DefaultNormalizer()
        self._granularity = granularity

    @property
    def granularity(self) -> str:
        return self._granularity

    def _splitter(self) -> Callable[[str], list[str]]:
        cap = self._registry.get("splitter", self._granularity)
        if cap is None:
            raise NLQLRegistryError(
                f"no splitter registered for granularity {self._granularity!r}"
            )
        return cast(Callable[[str], list[str]], cap.impl)

    def process(self, documents: Iterable[Document]) -> list[Unit]:
        """Normalize, split, and embed documents into units (vectors attached).

        Text payloads are split into segments; image payloads become one unit each and are
        embedded via the embedder's ``embed_images`` when it is multimodal (otherwise they
        are skipped). Both paths share the same index — only the vector's origin differs.
        """
        splitter = self._splitter()
        can_embed_images = supports_images(self._embedder)
        units: list[Unit] = []
        text_units: list[Unit] = []
        texts: list[str] = []
        image_units: list[Unit] = []
        images: list[bytes | str] = []

        for doc in documents:
            ordinal = 0
            for payload in doc.payloads:
                if payload.is_text:
                    for segment in splitter(self._normalizer.normalize(payload.as_text)):
                        unit = Unit(
                            id=f"{doc.id}#{self._granularity}:{ordinal}",
                            doc_id=doc.id,
                            kind=self._granularity,  # type: ignore[arg-type]
                            payload=Payload.text(segment),
                            metadata=dict(doc.metadata),  # copy so units never share/mutate
                            ordinal=ordinal,
                        )
                        units.append(unit)
                        text_units.append(unit)
                        texts.append(segment)
                        ordinal += 1
                elif payload.modality is Modality.IMAGE and can_embed_images:
                    unit = Unit(
                        id=f"{doc.id}#{self._granularity}:{ordinal}",
                        doc_id=doc.id,
                        kind=self._granularity,  # type: ignore[arg-type]
                        payload=payload,
                        metadata=dict(doc.metadata),
                        ordinal=ordinal,
                    )
                    units.append(unit)
                    image_units.append(unit)
                    images.append(payload.data)
                    ordinal += 1
                # else: unsupported modality → skipped

        if texts:
            for unit, vector in zip(text_units, self._embedder.embed(texts), strict=True):
                unit.vector = vector
        if images:
            image_vectors = self._embedder.embed_images(images)  # type: ignore[attr-defined]
            for unit, vector in zip(image_units, image_vectors, strict=True):
                unit.vector = vector
        return units
