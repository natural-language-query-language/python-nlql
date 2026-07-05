# 混合后端

NLQL 的查询与后端解耦：内置存储开箱即用，也可接入 Qdrant、Faiss 等专用向量库。同一查询在不同后端上结果一致。

## 一致性

所有后端遵循同一套 `Store` 接口。引擎优先用后端自带的能力处理过滤等操作（更快），后端无法处理的部分在内存中完成。无论走哪条路径，最终结果一致，仅性能存在差异。

## 示例

```python
from nlql import Document, Engine
from nlql.embed import FakeEmbedder
from nlql.store import LocalStore

CORPUS = [
    Document.from_text("Machine learning models learn patterns.",
                       id="d1", metadata={"status": "published", "year": 2024}),
    Document.from_text("Neural networks power deep learning.",
                       id="d2", metadata={"status": "published", "year": 2025}),
    Document.from_text("Banana bread needs flour and sugar.",
                       id="d3", metadata={"status": "draft", "year": 2024}),
    Document.from_text("Reinforcement learning trains agents.",
                       id="d4", metadata={"status": "published", "year": 2020}),
]

QUERY = """
    SELECT SENTENCE
    LET rel = SIMILARITY(content, "deep learning networks")
    WHERE meta.status == "published" AND meta.year >= 2024
    ORDER BY rel DESC
    LIMIT 3
"""


def backends():
    stores = {"LocalStore": LocalStore()}
    try:
        from nlql.store.faiss_store import FaissStore
        stores["FaissStore"] = FaissStore()
    except Exception:
        print("(未安装 faiss，跳过)")
    try:
        from nlql.store.qdrant_store import QdrantStore
        stores["QdrantStore"] = QdrantStore()
    except Exception:
        print("(未安装 qdrant，跳过)")
    return stores


for name, store in backends().items():
    engine = Engine(FakeEmbedder(dim=64), store=store)
    engine.add_documents(CORPUS)
    hits = [(u.doc_id, round(u.scores["rel"], 3)) for u in engine.search(QUERY)]
    print(f"{name:12} → {hits}")
```

各后端的 `hits` 完全相同（`d2`、`d1`，分数一致）。

## 后端能力

| 后端 | 向量检索 | 元数据过滤 | 全文搜索 | 安装 |
|---|---|---|---|---|
| `LocalStore` | 内置（精确） | 内置 | 内存 | 内置 |
| `FaissStore` | Faiss（精确） | 内存 | 内存 | `nlql[faiss]` |
| `HnswStore` | HnswLib（近似，适合大数据量） | 过取 + 内存 | 内存 | `nlql[hnsw]` |
| `QdrantStore` | Qdrant | 原生 | 内存 | `nlql[qdrant]` |
| `ChromaStore` | Chroma | 原生 | 内存 | `nlql[chroma]` |
| `PgVectorStore` | Postgres + pgvector | 原生 | 原生（ILIKE） | `nlql[pgvector]` |

!!! note "默认后端"
    不传入 `store` 时使用 `LocalStore`——纯 Python，适合万级到十万级数据量。数据量更大时切换 `HnswStore` 或 `FaissStore`。

!!! tip "切换后端"
    ```python
    from nlql.store.qdrant_store import QdrantStore
    engine = nlql.Engine(embedder, store=QdrantStore(location=":memory:"))
    ```
    写入与查询代码无需改动。

## 下一步

- [Store API](../../reference/store.md)
- [快速开始](quickstart.md)
- [性能](../../performance.md)
- [设计思路](../concepts/overview.md)

---

**完整源码**：[`examples/hybrid_stores.py`](https://github.com/natural-language-query-language/python-nlql/blob/main/examples/hybrid_stores.py)（需 `pip install "python-nlql[faiss,qdrant]"`）
