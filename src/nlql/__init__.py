"""NLQL — SQL-style semantic query language and retrieval middleware for Agents & RAG.

Quick start::

    from nlql import Engine
    from nlql.embed import FakeEmbedder   # or OpenAIEmbedder(base_url=..., api_key=...)

    engine = Engine(FakeEmbedder())
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

Backend implementations (embedders, stores, concrete rerankers) live in their
submodules: `from nlql.embed import OpenAIEmbedder`, `from nlql.store import
LocalStore`, `from nlql.rerank import CrossEncoderReranker`.
"""

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
from nlql.rerank import Reranker
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
from nlql.types import Signature, TypeTag, TypeHandler, register_type, get_type_handler

__version__ = "0.3.3"

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
    # rerank protocol (concrete rerankers in nlql.rerank)
    "Reranker",
    # registry
    "Registry",
    "GLOBAL_REGISTRY",
    "register_function",
    "register_splitter",
    # types
    "Signature",
    "TypeTag",
    "TypeHandler",
    "register_type",
    "get_type_handler",
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
