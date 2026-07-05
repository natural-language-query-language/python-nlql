"""Built-in text splitters, registered as ``splitter`` capabilities.

The defaults are rule-based and language-agnostic-ish: Western terminal punctuation
(``. ! ?``), CJK punctuation (``。！？…``), and newlines act as boundaries. This is a
pragmatic baseline — for production-grade segmentation users register a better splitter
(pysbd / spaCy / jieba / nltk) under the same ``SENTENCE`` name, optionally routed by
language. Rule-based splitting will over-split abbreviations like "U.S.A."; that is the
documented trade-off of the zero-dependency default.
"""

from __future__ import annotations

import re

from nlql.registry.core import GLOBAL_REGISTRY

# Split *after* a run of terminal punctuation: Western needs trailing whitespace to avoid
# breaking decimals; CJK punctuation may be followed directly by the next character.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|(?<=[。！？…])\s*|\n+")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using punctuation and newline boundaries."""
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_BOUNDARY.split(text)
    return [p.strip() for p in parts if p and p.strip()]


def split_chunks(text: str, max_chars: int = 1000) -> list[str]:
    """Group whole sentences into chunks no longer than ``max_chars`` characters."""
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    current = ""
    for sentence in split_sentences(text):
        if current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip() if current else sentence
    if current:
        chunks.append(current)
    return chunks or [text]


# Seed the defaults. Names match SELECT units (case-insensitive lookup).
GLOBAL_REGISTRY.register("splitter", "SENTENCE", split_sentences, doc="rule-based sentence splitter")
GLOBAL_REGISTRY.register("splitter", "CHUNK", split_chunks, doc="size-bounded chunk splitter")
