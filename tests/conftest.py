"""Shared pytest fixtures.

The deterministic :class:`FakeEmbedder` keeps the whole semantic pipeline testable
offline — no model download, no network, stable vectors across processes.
"""

from __future__ import annotations

import pytest

from nlql.embed import FakeEmbedder


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder(dim=64)
