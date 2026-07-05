[根目录 CLAUDE.md](../../CLAUDE.md) > [src](../) > [nlql](.) > **sdk**

# sdk — 高层 SDK：Engine + Query Builder

## 模块职责

面向使用者的入口层，**全部编译到 Query IR**。包含两类入口：

- **`Engine`**：检索引擎，主入口；统一 ingestion API 与查询路径；纯依赖注入构造。
- **`QueryBuilder`**（`builder.py`）：流式、类型安全的 Python 入口，与 NLQL 字符串 / LLM IR 等价。

## 入口与启动

```python
import nlql
engine = nlql.Engine(nlql.FakeEmbedder())        # 或 OpenAIEmbedder(base_url=..., api_key=...)
engine.add_text("...", metadata={"status": "published"})
results = engine.search('SELECT SENTENCE LET rel = SIMILARITY(content, "x") WHERE rel >= 0.2 LIMIT 5')
```

- 模块入口：`src/nlql/sdk/__init__.py` 导出 `Engine`、`QueryBuilder` 及构造工具 `select / F / Meta / content / field / similarity / contains / length`。
- 通过根包 `import nlql` 即可拿到全部 public 名字。

## 对外接口（`Engine`，依赖注入）

```python
Engine(
    embedder: Embedder,                 # 唯一必填，embedder 即扩展点
    *,
    store: Store | None = None,         # 注入任意后端；None → LocalStore()
    registry: Registry | None = None,   # None → GLOBAL_REGISTRY.child()
    granularity: str = "sentence",
    normalizer: Normalizer | None = None,
    cache: EmbeddingCache | None = None,
    field_types: dict[str, TypeTag] | None = None,   # 声明式 meta 类型
    reranker: Reranker | None = None,   # 两段式检索的精排器
    rerank_factor: int = 5,             # 召回过取倍数
    named_embedders: dict[str, Embedder] | None = None,  # named 多向量
)
```

- 写入：`add` / `add_text` / `add_image`（多模态）/ `add_file` / `add_files` / `add_documents` / `add_multivector`。
- 查询：`search(str | Query)`、`explain(...)`、`search_ir(dict)`、`function_schema()`、`function_tool()`。
- 扩展：`register_function(...)`、`register_splitter(...)`（实例作用域，shadow 全局）。
- 访问器：`.store` / `.registry` / `.granularity` / `__len__`。

> 注意 `store if store is not None else LocalStore()`——空 store 是 falsy，用 `or` 会被错误替换为 LocalStore。

## 关键依赖与配置

- 上游：`embed`（CachedEmbedder 包裹注入的 embedder）/ `ingest.IngestionPipeline` / `exec.Executor` / `lang.NLQLParser` / `loaders.load_documents` / `registry` / `store.LocalStore`。
- 核心：`numpy`。

## 数据模型

- 写入与查询都返回 / 操作 `model.Unit`（带 `.scores` 字典、`.vectors` 命名字典、`.payload` 模态无关）。
- 文档来源统一为 `model.Document`。

## 测试与质量

- `tests/test_sdk.py`（Engine 集成）、`tests/test_builder.py`（QueryBuilder round-trip）、`tests/test_multivector.py`（named 向量）。
- 三入口等价性测试是本模块的硬契约。

## 常见问题 (FAQ)

- **如何换渠道？** `Engine(OpenAIEmbedder(base_url="https://your-gateway/v1"))`；其它厂商 = 一个新 `Embedder` 实现。**不要**给 Engine 加 `with_xxx()` 工厂。
- **多模态怎么用？** `Engine(FakeMultimodalEmbedder(), granularity="chunk")` + `engine.add_image(bytes|path|url)`。
- **如何挂外部库？** `Engine(embedder, store=QdrantStore(location=":memory:"))`。

## 相关文件清单

- [`__init__.py`](__init__.py) — public 导出
- [`engine.py`](engine.py) — `Engine` 类（主入口）
- [`builder.py`](builder.py) — `QueryBuilder` 与表达式构造工具

## 变更记录

- 2026-07-05：首次生成。
