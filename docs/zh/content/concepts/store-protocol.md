# Store 接口

## 一个接口，多个后端

`Store` 是存储与检索的统一接口。`Engine` 不关心数据落在哪里——内置的内存索引、一个 Faiss 索引、一个 Qdrant 集合、一张 Postgres 表，只要它们实现了同一套方法，引擎就能用同一份查询代码驱动它们。

```python
from __future__ import annotations
from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable
import numpy as np
from nlql.ir.nodes import Expr
from nlql.model import Document, Unit

@runtime_checkable
class Store(Protocol):
    def upsert(self, units: Sequence[Unit]) -> None: ...
    def add_documents(self, documents: Iterable[Document]) -> None: ...
    def get_document(self, doc_id: str) -> Document | None: ...
    def ann_search(
        self,
        vector: np.ndarray,
        k: int | None = None,
        *,
        filter: Expr | None = None,
    ) -> list[tuple[Unit, float]]: ...
    def scan(self, filter: Expr | None = None) -> list[Unit]: ...
    def all_units(self) -> list[Unit]: ...
    def neighbors(self, doc_id: str, ordinal: int, window: int) -> list[Unit]: ...
    def capabilities(self) -> StoreCaps: ...
    def __len__(self) -> int: ...
```

两类方法各司其职：`upsert` / `add_documents` 负责写入，`ann_search` / `scan` / `neighbors` 负责查询。`ann_search` 接收一个 `filter` 参数——它是 IR 数据（`WHERE` 子表达式），不是编译好的 Python 谓词，因此每个后端都能把它翻译成自己的原生查询语法。

## StoreCaps：声明后端能做什么

后端之间的能力并不相同。有的支持近似最近邻，有的只做精确检索；有的能在自己的查询引擎里过滤元数据，有的不行；有的能原生处理全文 `CONTAINS`，有的不能。`StoreCaps` 把这些差异显式声明出来，供 Planner 决定哪些条件交给后端、哪些留作内存后置。

```python
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class StoreCaps:
    name: str = "local"
    vector_search: bool = True
    exact: bool = True               # 精确(flat) 还是近似(ANN) 召回
    metadata_pushdown: bool = False  # 能否在自己的查询引擎里过滤元数据
    text_pushdown: bool = False      # 能否原生处理 CONTAINS（如 SQL ILIKE）
```

后端返回的 `StoreCaps` 必须诚实。声明了 `metadata_pushdown=True`，就意味着你承诺在 `ann_search` / `scan` 里把元数据过滤翻译成后端原生查询、由后端完成。做不到就不要声明——引擎会在内存里补上，结果仍然正确。

## 六个后端的能力矩阵

| 后端 | 向量检索 | 元数据原生过滤 | 全文 `CONTAINS` 原生 | 安装 |
|---|---|---|---|---|
| `LocalStore` | numpy 精确点积 | 是（numpy 掩码） | 内存 | 内置 |
| `FaissStore` | Faiss 精确 | 否（内存后置） | 内存 | `pip install "python-nlql[faiss]"` |
| `HnswStore` | hnswlib 近似 | 是（过取 + 内存） | 内存 | `pip install "python-nlql[hnsw]"` |
| `QdrantStore` | Qdrant 近似 | 是（原生 Filter） | 内存 | `pip install "python-nlql[qdrant]"` |
| `ChromaStore` | Chroma 近似 | 是（原生 `where`） | 内存 | `pip install "python-nlql[chroma]"` |
| `PgVectorStore` | Postgres + pgvector | 是（SQL `WHERE`） | 是（`ILIKE`） | `pip install "python-nlql[pgvector]"` |

跨内置 / Faiss / HnswLib / Qdrant / Chroma 五个后端，同一句查询返回的候选与分数逐条相同（由跨后端测试保证）。PgVector 额外把 `CONTAINS(content, "x")` 翻译成 `content ILIKE '%x%'`，在数据库侧完成全文匹配。

## 写一个自定义 Store

接一个新后端，就是实现 `Store` 协议。共享逻辑可以从 `BaseUnitStore` 继承——它统一处理"过取候选 → 内存后置过滤"的回退路径，让你只需关心后端能做的部分。

```python
from nlql.store.common import BaseUnitStore
from nlql.store.base import StoreCaps

class MyBackendStore(BaseUnitStore):
    def capabilities(self) -> StoreCaps:
        return StoreCaps(
            name="my-backend",
            vector_search=True,
            exact=False,            # 近似 ANN
            metadata_pushdown=True, # 我把元数据过滤翻译成后端查询
            text_pushdown=False,
        )

    def ann_search(self, vector, k=None, *, filter=None):
        native_filter = self._translate(filter)     # IR → 后端原生语法
        rows = self._backend.query(vector, k or 100, native_filter)
        return [(self._to_unit(r), float(r.score)) for r in rows]
```

要点：

- `_translate` 把 IR 形式的 `filter` 翻成后端原生查询；这是你能把过滤交给后端的唯一方式。
- 你处理不了的条件留在 `filter` 里不动，`BaseUnitStore` 会在返回的候选上用内存逻辑再过一遍。
- `capabilities()` 必须与实际行为一致。声明能力但没在 `ann_search` 里落实，会导致结果错误；能做却不声明，只是少了一次性能优化，结果仍对。

## 下一步

- 这种能力拆分如何影响查询计划：见 [混合引擎](./hybrid-pushdown.md)
- 跨后端的可运行示例：见 [混合后端教程](../tutorials/hybrid-stores.md)
- Store 协议的方法签名细节：见 [Store 参考页](../../reference/store.md)
