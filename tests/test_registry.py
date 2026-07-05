"""Tests for the unified capability registry."""

import pytest

from nlql.errors import NLQLRegistryError
from nlql.registry import GLOBAL_REGISTRY, Registry
from nlql.registry.core import Capability
from nlql.types import Signature, TypeTag


class TestBuiltins:
    def test_builtin_functions_seeded(self) -> None:
        for name in ("CONTAINS", "MATCH", "LIKE", "LENGTH", "COUNT", "LOWER", "UPPER", "SIMILARITY"):
            assert GLOBAL_REGISTRY.has("function", name), name

    def test_case_insensitive_lookup(self) -> None:
        assert GLOBAL_REGISTRY.get("function", "contains") is not None
        assert GLOBAL_REGISTRY.get("function", "Contains") is not None
        assert GLOBAL_REGISTRY.get("function", "CONTAINS") is not None

    def test_similarity_is_provider_backed(self) -> None:
        cap = GLOBAL_REGISTRY.get("function", "SIMILARITY")
        assert cap is not None
        assert cap.provides_score is True
        assert cap.impl is None
        assert cap.signature is not None
        assert cap.signature.returns is TypeTag.NUMBER

    def test_builtin_impls(self) -> None:
        contains = GLOBAL_REGISTRY.get("function", "CONTAINS").impl
        like = GLOBAL_REGISTRY.get("function", "LIKE").impl
        count = GLOBAL_REGISTRY.get("function", "COUNT").impl
        length = GLOBAL_REGISTRY.get("function", "LENGTH").impl
        assert contains("Hello World", "world") is True
        assert contains("Hello", "xyz") is False
        assert like("draft-2024", "draft%") is True
        assert like("published", "draft%") is False
        assert like("a1c", "a_c") is True
        assert count("na na na", "na") == 3
        assert length("hello") == 5


class TestRegistration:
    def test_register_and_get(self) -> None:
        reg = Registry()
        reg.register("function", "MY_FN", lambda x: x, signature=Signature((TypeTag.ANY,), TypeTag.ANY))
        cap = reg.get("function", "my_fn")
        assert isinstance(cap, Capability)
        assert cap.name == "MY_FN"

    def test_decorator_registration(self) -> None:
        reg = Registry()

        @reg.function("WORD_COUNT", signature=Signature((TypeTag.TEXT,), TypeTag.NUMBER))
        def wc(text: str) -> int:
            return len(text.split())

        assert reg.get("function", "WORD_COUNT").impl("a b c") == 3

    def test_unknown_kind_rejected(self) -> None:
        reg = Registry()
        with pytest.raises(NLQLRegistryError):
            reg.register("gadget", "X", lambda: None)

    def test_empty_name_rejected(self) -> None:
        reg = Registry()
        with pytest.raises(NLQLRegistryError):
            reg.register("function", "  ", lambda: None)

    def test_duplicate_without_overwrite_rejected(self) -> None:
        reg = Registry()
        reg.register("function", "DUP", lambda: 1)
        with pytest.raises(NLQLRegistryError):
            reg.register("function", "DUP", lambda: 2)

    def test_duplicate_with_overwrite_allowed(self) -> None:
        reg = Registry()
        reg.register("function", "DUP", lambda: 1)
        reg.register("function", "DUP", lambda: 2, overwrite=True)
        assert reg.get("function", "DUP").impl() == 2


class TestScopeChain:
    def test_child_resolves_parent(self) -> None:
        parent = Registry()
        parent.register("function", "SHARED", lambda: "parent")
        child = parent.child()
        assert child.get("function", "SHARED").impl() == "parent"

    def test_child_shadows_parent_without_overwrite(self) -> None:
        parent = Registry()
        parent.register("function", "SCORE", lambda: "global")
        child = parent.child()
        # Shadowing a parent entry is NOT a clash — no overwrite needed.
        child.register("function", "SCORE", lambda: "instance")
        assert child.get("function", "SCORE").impl() == "instance"
        # Parent is untouched.
        assert parent.get("function", "SCORE").impl() == "global"

    def test_names_merge_with_shadowing(self) -> None:
        parent = Registry()
        parent.register("function", "A", lambda: None)
        parent.register("function", "B", lambda: None)
        child = parent.child()
        child.register("function", "B", lambda: None)  # shadow
        child.register("function", "C", lambda: None)
        assert child.names("function") == ["A", "B", "C"]

    def test_instance_registry_isolation(self) -> None:
        # Two independent instance registries over the same global do not leak.
        r1 = GLOBAL_REGISTRY.child()
        r2 = GLOBAL_REGISTRY.child()
        r1.register("function", "ONLY_R1", lambda: 1)
        assert r1.has("function", "ONLY_R1")
        assert not r2.has("function", "ONLY_R1")
        # But both still see the global built-ins.
        assert r1.has("function", "SIMILARITY")
        assert r2.has("function", "SIMILARITY")
