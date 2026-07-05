"""The high-level Engine — the primary entry point for applications.

Fixes the reference implementation's awkward surface: ingestion is unified on the engine
(``add`` / ``add_text`` / ``add_documents``) rather than buried in one adapter, and every
query path (string, Query Builder, LLM IR) runs through the same executor. Embedding is
always cached, so nothing is ever embedded twice.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

from nlql.embed import CachedEmbedder, EmbeddingCache
from nlql.embed.base import Embedder
from nlql.errors import NLQLError
from nlql.exec import Executor
from nlql.ingest import IngestionPipeline
from nlql.ingest.normalize import Normalizer
from nlql.ir import Query, query_json_schema
from nlql.lang import NLQLParser
from nlql.loaders import Loader, load_documents
from nlql.model import Document, Modality, Payload, Unit
from nlql.registry import GLOBAL_REGISTRY, Registry
from nlql.rerank.base import Reranker
from nlql.store import LocalStore, Store
from nlql.types import Signature, TypeTag


class Engine:
    """A retrieval engine: ingest documents, run NLQL queries, extend capabilities.

    The embedder is injected — it *is* the extensibility point, so there are no
    provider-specific constructors to multiply. Any OpenAI-compatible channel is one
    :class:`~nlql.embed.OpenAIEmbedder` parameterized by ``base_url``; any other provider
    is just another :class:`~nlql.embed.base.Embedder` implementation::

        from nlql import Engine
        from nlql.embed import FakeEmbedder, OpenAIEmbedder

        Engine(OpenAIEmbedder(base_url="https://your-gateway/v1", api_key="sk-..."))
        Engine(FakeEmbedder())                       # deterministic, offline (tests/demos)
        Engine(MyCohereEmbedder(...))                # your own backend, no core change
    """

    def __init__(
        self,
        embedder: Embedder,
        *,
        store: Store | None = None,
        registry: Registry | None = None,
        granularity: str = "chunk",
        normalizer: Normalizer | None = None,
        cache: EmbeddingCache | None = None,
        field_types: dict[str, TypeTag] | None = None,
        reranker: Reranker | None = None,
        rerank_factor: int = 5,
        named_embedders: dict[str, Embedder] | None = None,
    ) -> None:
        self._registry = registry if registry is not None else GLOBAL_REGISTRY.child()
        self._embedder: Embedder = (
            embedder if isinstance(embedder, CachedEmbedder) else CachedEmbedder(embedder, cache)
        )
        # Named vectors (SIMILARITY(vec.<name>, …)) each use their own embedder, cached too.
        self._named_embedders: dict[str, Embedder] = {
            name: (e if isinstance(e, CachedEmbedder) else CachedEmbedder(e))
            for name, e in (named_embedders or {}).items()
        }
        # NOTE: `is not None`, not `or` — an empty store is falsy (len 0) and `or` would
        # silently swap in a LocalStore, ignoring the injected backend.
        self._store: Store = store if store is not None else LocalStore()
        self._granularity = granularity
        self._pipeline = IngestionPipeline(
            self._embedder, registry=self._registry, normalizer=normalizer, granularity=granularity
        )
        self._executor = Executor(
            self._store,
            self._registry,
            self._embedder,
            granularity=granularity,
            field_types=field_types,
            reranker=reranker,
            rerank_factor=rerank_factor,
            named_embedders=self._named_embedders,
        )
        self._parser = NLQLParser()

    # -- ingestion -------------------------------------------------------------
    def add(self, document: Document) -> None:
        """Ingest a single document."""
        self.add_documents([document])

    def add_text(
        self,
        text: str,
        *,
        id: str | None = None,
        metadata: dict[str, Any] | None = None,
        source: str | None = None,
    ) -> str:
        """Ingest a single text; returns the (generated or provided) document id."""
        doc_id = id or uuid.uuid4().hex
        self.add(Document.from_text(text, id=doc_id, metadata=metadata, source=source))
        return doc_id

    def add_image(
        self,
        image: bytes | str,
        *,
        id: str | None = None,
        metadata: dict[str, Any] | None = None,
        source: str | None = None,
        mime: str | None = None,
    ) -> str:
        """Ingest a single image as a multimodal document; returns its id.

        ``image`` may be raw ``bytes``, a local file path, or an ``http(s)`` / ``data:`` URL.
        Needs a multimodal embedder (e.g. ``DoubaoVisionEmbedder`` or ``ClipEmbedder``) — the
        image is embedded into the same space as text, so text queries retrieve it. Use
        ``granularity="chunk"`` for image collections.
        """
        if isinstance(image, (bytes, bytearray)):
            data: bytes | str = bytes(image)
        elif image.startswith(("http://", "https://", "data:")):
            data = image  # URL / data URI — the embedder fetches or decodes it
        else:
            data = Path(image).read_bytes()  # local file path → bytes
        doc_id = id or uuid.uuid4().hex
        payload = Payload(modality=Modality.IMAGE, data=data, mime=mime)
        self.add(Document(id=doc_id, payloads=[payload], metadata=metadata or {}, source=source))
        return doc_id

    def add_file(
        self,
        path: str,
        *,
        metadata: dict[str, Any] | None = None,
        loader: Loader | None = None,
    ) -> list[str]:
        """Load a file (``.txt`` / ``.md`` / ``.docx`` / ``.pdf`` / …) and ingest it.

        Dispatches by extension (override with ``loader=``). Returns the document ids. Needs
        the relevant extra for docx/pdf: ``pip install python-nlql[loaders]``.
        """
        documents = load_documents(path, loader=loader, metadata=metadata)
        self.add_documents(documents)
        return [doc.id for doc in documents]

    def add_files(self, paths: Iterable[str], *, metadata: dict[str, Any] | None = None) -> list[str]:
        """Load and ingest multiple files; returns all document ids."""
        ids: list[str] = []
        for path in paths:
            ids.extend(self.add_file(path, metadata=metadata))
        return ids

    def add_multivector(
        self,
        id: str,
        *,
        content: str,
        named: dict[str, str | bytes],
        metadata: dict[str, Any] | None = None,
        kind: str | None = None,
    ) -> str:
        """Ingest one record with several named vectors, each queryable via ``vec.<name>``.

        ``content`` is the record's text (and its default vector, via the primary embedder).
        ``named`` maps a vector name to its source — a ``str`` (embedded as text) or image
        ``bytes`` (embedded via that embedder's ``embed_images``). Configure per-name embedders
        with ``Engine(..., named_embedders={"image": ClipEmbedder(), ...})``. Query a specific
        vector with ``SIMILARITY(vec.image, "…")``.
        """
        unit = Unit(
            id=id,
            doc_id=id,
            kind=kind or self._granularity,  # type: ignore[arg-type]
            payload=Payload.text(content),
            metadata=dict(metadata or {}),
            vector=self._embedder.embed([content])[0],
            ordinal=0,
        )
        for name, data in named.items():
            embedder = self._named_embedders.get(name)
            if embedder is None:
                raise NLQLError(f"no named embedder configured for {name!r}; pass named_embedders=")
            if isinstance(data, (bytes, bytearray)):
                embed_images = getattr(embedder, "embed_images", None)
                if not callable(embed_images):
                    raise NLQLError(f"named embedder {name!r} cannot embed image bytes")
                unit.vectors[name] = np.asarray(embed_images([data])[0], dtype=np.float32)
            else:
                unit.vectors[name] = np.asarray(embedder.embed([data])[0], dtype=np.float32)
        self._store.upsert([unit])
        return id

    def add_documents(self, documents: Iterable[Document], *, batch: int = 256) -> None:
        """Ingest documents in batches (normalize → split → embed(cache) → index)."""
        buffer: list[Document] = []
        for doc in documents:
            buffer.append(doc)
            if len(buffer) >= batch:
                self._flush(buffer)
                buffer = []
        if buffer:
            self._flush(buffer)

    def _flush(self, documents: list[Document]) -> None:
        units = self._pipeline.process(documents)
        self._store.upsert(units)
        self._store.add_documents(documents)

    # -- querying --------------------------------------------------------------
    def _as_query(self, query: str | Query) -> Query:
        return self._parser.parse(query) if isinstance(query, str) else query

    def search(
        self,
        query: str | Query,
        *,
        limit: int | None = None,
        reranker: Reranker | None = None,
        rerank_query: str | None = None,
    ) -> list[Unit]:
        """Run an NLQL string or Query IR and return matching units.

        A ``reranker`` (here or on the engine) refines the recalled candidates: recall
        over-fetches, then the reranker re-scores each ``(query, passage)`` pair and the top
        ``limit`` are returned. ``rerank_query`` overrides the text used (default: the primary
        ``SIMILARITY`` query).
        """
        ir = self._as_query(query)
        if limit is not None:
            ir.limit = limit
        return self._executor.execute(ir, reranker=reranker, rerank_query=rerank_query)

    def explain(self, query: str | Query) -> dict[str, Any]:
        """Return the query plan (parsed IR, scores, filter, recall strategy)."""
        plan = self._executor.plan(self._as_query(query)).explain()
        if self._executor.reranker is not None:
            plan["rerank"] = type(self._executor.reranker).__name__
        return plan

    # -- extensibility ---------------------------------------------------------
    def register_function(
        self,
        name: str,
        *,
        signature: Signature | None = None,
        provides_score: bool = False,
        pushdownable: bool = False,
        overwrite: bool = False,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register a function for this engine only (instance-scoped)."""
        return self._registry.function(
            name,
            signature=signature,
            provides_score=provides_score,
            pushdownable=pushdownable,
            overwrite=overwrite,
        )

    def register_splitter(
        self, name: str, *, overwrite: bool = False
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register/override a splitter for this engine only."""
        return self._registry.splitter(name, overwrite=overwrite)

    # -- LLM integration -------------------------------------------------------
    def function_schema(self) -> dict[str, Any]:
        """JSON Schema for the Query IR, with Call names constrained to this engine."""
        return query_json_schema(self._registry.names("function"))

    def function_tool(
        self, name: str = "nlql_query", description: str | None = None
    ) -> dict[str, Any]:
        """An OpenAI-style tool definition an LLM can call to emit a Query IR."""
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description
                or "Build a semantic retrieval query as an NLQL Query IR document.",
                "parameters": self.function_schema(),
            },
        }

    def search_ir(self, ir: dict[str, Any], *, limit: int | None = None) -> list[Unit]:
        """Execute a Query IR document (e.g. produced by an LLM via function-calling)."""
        return self.search(Query.from_dict(ir), limit=limit)

    # -- accessors -------------------------------------------------------------
    @property
    def store(self) -> Store:
        return self._store

    @property
    def registry(self) -> Registry:
        return self._registry

    @property
    def granularity(self) -> str:
        return self._granularity

    def __len__(self) -> int:
        return len(self._store)
