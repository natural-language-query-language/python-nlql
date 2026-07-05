"""Language detection and routing for pluggable, robust segmentation.

The built-in rule splitter already handles CJK and Western punctuation. For production-grade
boundaries (abbreviations like "U.S.A.", locale rules) register a better backend under the
``SENTENCE`` splitter name, optionally wrapped in a :class:`LanguageRouter` so each language
uses the right engine. ``make_pysbd_splitter`` provides one such backend.
"""

from __future__ import annotations

from collections.abc import Callable

from nlql.errors import NLQLError

Splitter = Callable[[str], list[str]]


def detect_language(text: str) -> str:
    """Coarse script detection: ``"cjk"`` if the text is substantially CJK, else ``"latin"``."""
    cjk = sum(1 for ch in text if "　" <= ch <= "鿿" or "＀" <= ch <= "￯")
    total = sum(1 for ch in text if not ch.isspace())
    return "cjk" if total and cjk / total > 0.2 else "latin"


class LanguageRouter:
    """A splitter that dispatches to a per-language sub-splitter by detected script."""

    def __init__(self, routes: dict[str, Splitter], default: Splitter) -> None:
        self._routes = dict(routes)
        self._default = default

    def __call__(self, text: str) -> list[str]:
        return self._routes.get(detect_language(text), self._default)(text)


def make_pysbd_splitter(language: str = "en") -> Splitter:
    """A robust sentence splitter backed by pysbd (handles abbreviations).

    Needs ``pip install pysbd``. Register it as ``SENTENCE`` (globally or per engine) to
    replace the rule-based default.
    """
    try:
        import pysbd
    except ImportError as e:  # pragma: no cover - exercised only without pysbd
        raise NLQLError("pysbd not installed: pip install pysbd") from e

    segmenter = pysbd.Segmenter(language=language, clean=False)

    def split(text: str) -> list[str]:
        return [s.strip() for s in segmenter.segment(text) if s and s.strip()]

    return split
