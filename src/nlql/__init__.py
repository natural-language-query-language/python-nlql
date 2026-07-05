"""NLQL — SQL-style semantic query language and retrieval middleware for Agents & RAG.

Quick start::

    import nlql

    # The embedder is injected — any OpenAI-compatible channel is one class + a base_url.
    engine = nlql.Engine(nlql.FakeEmbedder())   # or Engine(OpenAIEmbedder(base_url=..., api_key=...))
    engine.add_text("AI agents plan, use memory, and call tools.", metadata={"status": "published"})

    results = engine.search('''
        SELECT SENTENCE
        LET rel = SIMILARITY(content, "autonomous agents")
        WHERE rel >= 0.2 AND meta.status == "published"
        ORDER BY rel DESC
        LIMIT 5
    ''')
    for unit in results:
        print(unit.scores.get("rel"), unit.content)
"""

from nlql.embed import CachedEmbedder, EmbeddingCache, FakeEmbedder, OpenAIEmbedder
from nlql.errors import (
    NLQLError,
    NLQLExecutionError,
    NLQLParseError,
    NLQLPlanError,
    NLQLRegistryError,
    NLQLSchemaError,
    NLQLTypeError,
)
from nlql.ir import Query, query_json_schema
from nlql.lang import parse
from nlql.model import Document, Modality, Payload, Unit
from nlql.registry import GLOBAL_REGISTRY, Registry, register_function, register_splitter
from nlql.rerank import CrossEncoderReranker, FakeReranker, Reranker
from nlql.sdk import (
    Engine,
    F,
    Meta,
    QueryBuilder,
    contains,
    content,
    field,
    length,
    select,
    similarity,
)
from nlql.store import LocalStore
from nlql.types import Signature, TypeTag

__version__ = "0.2.0"

__all__ = [
    # engine & builder
    "Engine",
    "QueryBuilder",
    "select",
    "F",
    "Meta",
    "content",
    "field",
    "similarity",
    "contains",
    "length",
    # model
    "Document",
    "Payload",
    "Unit",
    "Modality",
    # language & IR
    "parse",
    "Query",
    "query_json_schema",
    # embedders
    "FakeEmbedder",
    "OpenAIEmbedder",
    "EmbeddingCache",
    "CachedEmbedder",
    # rerank
    "Reranker",
    "FakeReranker",
    "CrossEncoderReranker",
    # store & registry
    "LocalStore",
    "Registry",
    "GLOBAL_REGISTRY",
    "register_function",
    "register_splitter",
    # types
    "Signature",
    "TypeTag",
    # errors
    "NLQLError",
    "NLQLParseError",
    "NLQLSchemaError",
    "NLQLRegistryError",
    "NLQLTypeError",
    "NLQLPlanError",
    "NLQLExecutionError",
    "__version__",
]
