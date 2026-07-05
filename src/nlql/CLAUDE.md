[根目录 CLAUDE.md](../../CLAUDE.md) > [src](../) > **nlql**（包总览）

# nlql — 包总览

`nlql` 是单包（`src/nlql`），13 个子模块见 [根 CLAUDE.md 模块索引](../../CLAUDE.md#模块索引)。本文件给出**包级 public API** 与单文件模块 **errors** 的说明。

## 包级 public API（`__init__.py`）

`import nlql` 一次性导出：

- **Engine & Builder**：`Engine`, `QueryBuilder`, `select`, `F`, `Meta`, `content`, `field`, `similarity`, `contains`, `length`
- **model**：`Document`, `Payload`, `Unit`, `Modality`
- **language & IR**：`parse`, `Query`, `query_json_schema`
- **embedders**：`FakeEmbedder`, `OpenAIEmbedder`, `EmbeddingCache`, `CachedEmbedder`
- **rerank**：`Reranker`, `FakeReranker`, `CrossEncoderReranker`
- **store & registry**：`LocalStore`, `Registry`, `GLOBAL_REGISTRY`, `register_function`, `register_splitter`
- **types**：`Signature`, `TypeTag`
- **errors**：见下
- `__version__ = "0.2.0"`

## errors — 类型化异常层级（单文件模块 `errors.py`）

定义全栈的类型化异常，根 `NLQLError`，让调用方能按错误阶段精确捕获。

```python
from nlql import NLQLError, NLQLParseError, NLQLSchemaError, NLQLRegistryError, NLQLTypeError, NLQLPlanError, NLQLExecutionError
```

层级（按命名 + DESIGN 语义推断）：

```
NLQLError                  # 根
├── NLQLParseError         # lang: NLQL 字符串解析失败
├── NLQLSchemaError        # ir: JSON Schema / IR 结构不符
├── NLQLRegistryError      # registry: 未注册 / 重复注册 / arity 不符
├── NLQLTypeError          # types/exec: 类型规约 / 比较类型不合法
├── NLQLPlanError          # plan: 不可下推冲突 / 计划不可行
└── NLQLExecutionError     # exec: 执行期失败（Store 异常等）
```

- 无外部依赖；被所有模块在对应失败点抛出。
- 各模块 `tests/test_*.py` 用 `pytest.raises(...)` 验证（`test_lang.py` 验 parse 错、`test_registry.py` 验注册错 等）。
- 文件：[`errors.py`](errors.py)

## 子模块导航

进入各子模块的 `CLAUDE.md` 查看细节：[sdk](sdk/CLAUDE.md) · [lang](lang/CLAUDE.md) · [ir](ir/CLAUDE.md) · [embed](embed/CLAUDE.md) · [ingest](ingest/CLAUDE.md) · [store](store/CLAUDE.md) · [plan](plan/CLAUDE.md) · [exec](exec/CLAUDE.md) · [rerank](rerank/CLAUDE.md) · [loaders](loaders/CLAUDE.md) · [model](model/CLAUDE.md) · [registry](registry/CLAUDE.md) · [types](types/CLAUDE.md)

## 变更记录

- 2026-07-05：首次生成。包总览 + errors 单文件模块说明。
