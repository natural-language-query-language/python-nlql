"""Text normalization.

Pluggable via the engine. The default collapses inline whitespace and blank lines but
**preserves single newlines**, since the sentence splitter treats them as boundaries.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Protocol, runtime_checkable


@runtime_checkable
class Normalizer(Protocol):
    def normalize(self, text: str) -> str: ...


class DefaultNormalizer:
    """NFC-normalize, collapse spaces/tabs and blank lines, trim each line."""

    _INLINE_WS = re.compile(r"[ \t 　]+")
    _MULTI_NL = re.compile(r"\n{2,}")

    def normalize(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        text = self._INLINE_WS.sub(" ", text)
        text = self._MULTI_NL.sub("\n", text)
        text = "\n".join(line.strip() for line in text.split("\n"))
        return text.strip()
