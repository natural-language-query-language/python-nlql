# NLQL

[![PyPI version](https://img.shields.io/pypi/v/python-nlql.svg?label=pypi)](https://pypi.org/project/python-nlql/)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-online-blue.svg)](https://natural-language-query-language.github.io/python-nlql/)

**English** · [简体中文](README.zh-CN.md) · [在线文档](https://natural-language-query-language.github.io/python-nlql/)

NLQL lets you do semantic search with SQL-style statements. Relevance scoring, filtering, and sorting live in one query — no more scattered embedding calls and post-processing code.

Built for Agent and RAG applications: the query itself is structured data, usable directly as an LLM tool-call payload.

## What it looks like

```python
import nlql

engine = nlql.Engine(nlql.FakeEmbedder())  # or OpenAIEmbedder, or any Embedder
engine.add_text("AI agents plan tasks and call tools.", metadata={"status": "published"})
engine.add_text("Banana bread needs flour and sugar.", metadata={"status": "draft"})

for unit in engine.search('''
    SELECT SENTENCE
    LET rel = SIMILARITY(content, "autonomous agents")
    WHERE rel >= 0.2 AND meta.status == "published"
    ORDER BY rel DESC
    LIMIT 5
'''):
    print(f"{unit.scores['rel']:+.3f}  {unit.content}")
```

The statement reads almost like SQL: `SELECT` sets the return granularity, `LET` computes relevance, `WHERE` filters, `ORDER BY` / `LIMIT` sort and cap.

## Features

- **One statement, full intent** — relevance, filtering, and sorting in one place, not scattered across business code
- **Three ways to write, identical results** — SQL statement, Python chained builder, or JSON IR; all compile to the same internal representation
- **Pluggable backends** — built-in store works out of the box; switch to Qdrant / Faiss / Chroma / HnswLib / pgvector with one line
- **Two-stage retrieval** — attach a reranker after recall for higher accuracy
- **Multimodal** — text and images share one vector space; retrieve images with text
- **Explainable** — `engine.explain()` prints the query plan

## Installation

```bash
pip install python-nlql
```

Optional extras:

| Command | Purpose |
|---|---|
| `pip install "python-nlql[faiss]"` | Faiss backend |
| `pip install "python-nlql[hnsw]"` | HnswLib backend (for large-scale data) |
| `pip install "python-nlql[qdrant]"` | Qdrant backend |
| `pip install "python-nlql[chroma]"` | Chroma backend |
| `pip install "python-nlql[pgvector]"` | Postgres + pgvector backend |
| `pip install "python-nlql[local]"` | local sentence-transformers / CLIP / cross-encoder |
| `pip install "python-nlql[loaders]"` | DOCX / PDF file loaders |

## Switching backends

One line; ingestion and query code stay the same:

```python
from nlql.store.qdrant_store import QdrantStore
engine = nlql.Engine(embedder, store=QdrantStore(location=":memory:"))
```

## Documentation

Full docs, tutorials, and API reference: **https://natural-language-query-language.github.io/python-nlql/en/**

- [Quick start](https://natural-language-query-language.github.io/python-nlql/en/content/tutorials/quickstart/)
- [Design](https://natural-language-query-language.github.io/python-nlql/en/content/concepts/overview/)
- [API reference](https://natural-language-query-language.github.io/python-nlql/en/reference/sdk/)
- [中文文档](https://natural-language-query-language.github.io/python-nlql/)

More examples in the [`examples/`](examples/) directory.

## License

[MIT](LICENSE)
