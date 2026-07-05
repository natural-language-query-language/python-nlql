[根目录 CLAUDE.md](../../CLAUDE.md) > [src](../) > [nlql](.) > **store**

# store — Store 协议 + LocalStore + 后端适配器

## 模块职责

数据与索引的持有者。`Store` 是抽象 Protocol，`StoreCaps` 声明能力；`LocalStore` 自持（numpy FlatIndex + 元数据列 + 倒排），外部库适配器用**自己后端的原生能力**实现 `ann_search + 过滤 + limit`——"下推"是适配器的**内部细节**。

> 这正是"抽象 Store 层、让实现直接对接 Qdrant/Chroma/PgVector"的落地方式；下推 = "把查询翻译成后端原生查询"，后端表达不了的部分在内存兜底。

## 入口与启动

```python
from nlql.store import LocalStore
from nlql.store.qdrant_store import QdrantStore
engine = nlql.Engine(embedder, store=QdrantStore(location=":memory:"))
```

模块入口 `src/nlql/store/__init__.py` 导出 `Store / StoreCaps / LocalStore / matches_filter`；适配器按需单独导入。

## 对外接口（`Store` Protocol + `StoreCaps`）

```python
class StoreCaps:          # 能力声明，驱动 Planner 下推
    name: str = "local"
    vector_search: bool = True
    exact: bool = True            # 精确(flat) vs 近似(ANN)
    metadata_pushdown: bool = False
    text_pushdown: bool = False   # 能否原生推 CONTAINS（如 SQL ILIKE）

class Store(Protocol):
    def upsert(self, units: Sequence[Unit]) -> None: ...
    def add_documents(self, documents: Iterable[Document]) -> None: ...
    def get_document(self, doc_id: str) -> Document | None: ...
    def ann_search(self, vector, k=None, *, filter: Expr | None = None) -> list[tuple[Unit, float]]: ...
    def scan(self, filter: Expr | None = None) -> list[Unit]: ...
    def all_units(self) -> list[Unit]: ...
    def neighbors(self, doc_id, ordinal, window) -> list[Unit]: ...
    def capabilities(self) -> StoreCaps: ...
    def __len__(self) -> int: ...
```

> **关键约定**：传给 Store 的 `filter` 是 **IR 数据**（WHERE 子表达式）而非编译好的 Python 闭包——这样每个适配器都能自主翻译。`LocalStore` 编成 numpy 掩码，QdrantStore→Qdrant Filter，PgVectorStore→SQL WHERE + `<=>`。

## 后端矩阵

| 后端 | 向量检索 | 元数据下推 | 全文(CONTAINS)下推 | 安装 |
|---|---|---|---|---|
| `LocalStore` | numpy FlatIndex（精确，列式过滤） | ✓ numpy 掩码 | 内存 | 内置 |
| `FaissStore` | faiss（精确） | ✗（残余内存） | 内存 | `nlql[faiss]` |
| `HnswStore` | hnswlib（ANN, **sublinear**） | ✓ 过取+内存 | 内存 | `nlql[hnsw]` |
| `QdrantStore` | Qdrant（ANN） | ✓ 原生 Filter | 内存 | `nlql[qdrant]` |
| `ChromaStore` | Chroma（ANN） | ✓ 原生 where | 内存 | `nlql[chroma]` |
| `PgVectorStore` | Postgres+pgvector | ✓ SQL | ✓ **ILIKE** | `nlql[pgvector]` |

**跨 Local/Faiss/Hnsw/Qdrant/Chroma 五后端同一查询结果逐条一致**（`tests/test_cross_store.py` 保证）。

## 关键依赖与配置

- 内核：`numpy`；适配器各自的 extras（faiss-cpu / hnswlib / qdrant-client / chromadb / psycopg+pgvector），全部**懒导入**，永不进入核心依赖。
- `BaseUnitStore`（`common.py`）：适配器共享基类，统一处理"过取 → 候选 → 内存残余过滤"。

## 数据模型

- `Unit`（`model`）携带向量、命名向量字典、metadata、scores；适配器只负责存取原语。

## 测试与质量

- `tests/test_store.py`（LocalStore）、`test_faiss_store.py`、`test_hnsw_store.py`、`test_qdrant_store.py`、`test_chroma_store.py`、`test_pgvector_store.py`（含 CONTAINS→ILIKE 翻译单测）、`test_cross_store.py`（五后端一致性）、`test_columns.py`（列式过滤与逐行结果相等）、`test_pushdown.py`。

## 常见问题 (FAQ)

- **加新后端？** 实现 `Store` Protocol + 返回正确的 `StoreCaps`；继承 `BaseUnitStore` 复用残余过滤逻辑。
- **下推何时发生？** Planner 按 `StoreCaps` 决定哪些 WHERE 子表达式下推，其余内存后置。能整条委托就纯委托、无拆分。

## 相关文件清单

- [`base.py`](base.py) — `Store` Protocol + `StoreCaps`
- [`local.py`](local.py) — `LocalStore`（numpy FlatIndex）
- [`common.py`](common.py) — `BaseUnitStore`（适配器共享逻辑）
- [`columns.py`](columns.py) — numpy 列式元数据过滤（M6 性能优化）
- [`filter.py`](filter.py) — `matches_filter`（IR 表达式求值）
- [`faiss_store.py`](faiss_store.py) / [`hnsw_store.py`](hnsw_store.py) / [`qdrant_store.py`](qdrant_store.py) / [`chroma_store.py`](chroma_store.py) / [`pgvector_store.py`](pgvector_store.py) — 5 个适配器

## 变更记录

- 2026-07-05：首次生成。
