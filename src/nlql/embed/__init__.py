"""Embedder backends and embedding cache.

``SentenceTransformerEmbedder`` is intentionally *not* imported here so that importing
``nlql.embed`` never triggers the heavy optional dependency; import it explicitly from
``nlql.embed.sentence_transformers`` when needed.
"""

from nlql.embed.base import BaseEmbedder, Embedder, normalize_rows
from nlql.embed.cache import CachedEmbedder, EmbeddingCache, cache_key
from nlql.embed.doubao import DoubaoVisionEmbedder
from nlql.embed.fake import FakeEmbedder
from nlql.embed.multimodal import FakeMultimodalEmbedder, MultimodalEmbedder, supports_images
from nlql.embed.openai import OpenAIEmbedder

__all__ = [
    "Embedder",
    "BaseEmbedder",
    "normalize_rows",
    "FakeEmbedder",
    "OpenAIEmbedder",
    "EmbeddingCache",
    "CachedEmbedder",
    "cache_key",
    "MultimodalEmbedder",
    "FakeMultimodalEmbedder",
    "DoubaoVisionEmbedder",
    "supports_images",
]
