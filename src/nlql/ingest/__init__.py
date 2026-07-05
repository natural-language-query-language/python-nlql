"""Ingestion: normalization, pluggable splitting, and the write-time pipeline.

Importing this package registers the default ``SENTENCE`` and ``CHUNK`` splitters.
"""

# Side effect: register default splitters into GLOBAL_REGISTRY.
from nlql.ingest import splitters as _splitters  # noqa: E402,F401
from nlql.ingest.language import LanguageRouter, detect_language, make_pysbd_splitter
from nlql.ingest.normalize import DefaultNormalizer, Normalizer
from nlql.ingest.pipeline import IngestionPipeline
from nlql.ingest.splitters import split_chunks, split_sentences

__all__ = [
    "IngestionPipeline",
    "Normalizer",
    "DefaultNormalizer",
    "split_sentences",
    "split_chunks",
    "detect_language",
    "LanguageRouter",
    "make_pysbd_splitter",
]
