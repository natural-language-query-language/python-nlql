"""Tests for multilingual segmentation and language routing (M4)."""

from __future__ import annotations

import pytest

from nlql.ingest import LanguageRouter, detect_language, split_sentences


class TestDetectLanguage:
    def test_latin(self) -> None:
        assert detect_language("Hello world, how are you today?") == "latin"

    def test_chinese(self) -> None:
        assert detect_language("你好，今天天气很好。") == "cjk"

    def test_japanese(self) -> None:
        assert detect_language("こんにちは、元気ですか。") == "cjk"

    def test_empty(self) -> None:
        assert detect_language("") == "latin"


class TestMultilingualSplit:
    def test_english(self) -> None:
        assert split_sentences("Hello world. How are you? Fine!") == [
            "Hello world.",
            "How are you?",
            "Fine!",
        ]

    def test_chinese(self) -> None:
        assert split_sentences("你好。今天天气不错！真的吗？") == [
            "你好。",
            "今天天气不错！",
            "真的吗？",
        ]

    def test_japanese(self) -> None:
        assert split_sentences("こんにちは。元気ですか？はい！") == [
            "こんにちは。",
            "元気ですか？",
            "はい！",
        ]


class TestLanguageRouter:
    def test_dispatch_by_script(self) -> None:
        router = LanguageRouter(
            routes={"cjk": lambda t: ["CJK"], "latin": lambda t: ["LATIN"]},
            default=lambda t: ["DEFAULT"],
        )
        assert router("你好世界，今天天气不错。") == ["CJK"]
        assert router("hello there, this is english text") == ["LATIN"]


class TestPysbd:
    def test_abbreviations_not_oversplit(self) -> None:
        pytest.importorskip("pysbd")
        from nlql.ingest import make_pysbd_splitter

        split = make_pysbd_splitter("en")
        # pysbd keeps "Dr." and "U.S.A." intact; the rule-based default would over-split.
        result = split("Dr. Smith moved to the U.S.A. last year. He is happy now.")
        assert len(result) == 2
        assert result[0].startswith("Dr. Smith")
