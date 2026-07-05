# NLQL — 面向 Agent / RAG 的语义查询语言

> 用一套 **SQL 风格、语义清晰**的查询语言（及等价的结构化 IR），在纯文本、自持索引与外部向量库之上，提供统一、可解释、高性能的语义检索。

NLQL 把 SQL 的确定性逻辑与向量检索的模糊语义能力融合到一条语句里，定位是 **Agent / RAG 的检索中间件**：查询的规范形态是可 JSON 序列化的 `Query IR`，NLQL 字符串、Python Query Builder、LLM function-calling 都编译到同一个 IR。

```sql
SELECT SPAN(SENTENCE, window => 1)
LET   relevance = SIMILARITY(content, "AI Agent architecture"),
      novelty   = SIMILARITY(content, "novel unpublished ideas")
WHERE relevance >= 0.8
  AND meta.status != "draft"
  AND (content CONTAINS "planning" OR content CONTAINS "memory")
ORDER BY relevance DESC, novelty DESC
LIMIT 5
```

## 设计要点

- **IR 优先**：三种入口等价编译到 `Query IR`，序列化即 function-calling 载体。
- **索引是一等公民**：向量写入时计算并落索引，查询走「召回 → 精排」，绝不全量重算。
- **语义正交**：字段路径 / 标量函数 / 谓词 / 逻辑组合语义正交，求值单一路径，无 special-case。
- **统一注册**：算子 / 函数 / 分词 / embedder / 模态走同一注册协议。
- **模态无关**：数据模型从第一天起对文本 / 图像 / 二进制通用；v1 落地文本。
- **可解释**：每个查询都能 `EXPLAIN`，产出查询计划与下推情况。

完整设计见 [`DESIGN.md`](DESIGN.md)。

## 安装

```bash
pip install python-nlql              # 核心：numpy + lark + httpx，处处可装
pip install "python-nlql[local]"     # 可选：本地 sentence-transformers / CLIP / cross-encoder
pip install "python-nlql[faiss]"     # 可选：Faiss 后端（另有 hnsw / qdrant / chroma / pgvector）
pip install "python-nlql[loaders]"   # 可选：DOCX / PDF 文档加载
```

## 导入文档：DOCX / PDF / TXT

```python
engine.add_file("report.docx")                 # 按扩展名自动分派加载器
engine.add_files(["a.pdf", "b.md", "c.txt"])   # 批量；返回文档 id
# 扩展：register_loader(MyEpubLoader(), ".epub")；或 add_file(path, loader=PdfLoader(by_page=True))
```

## 存储后端（可插拔）

`Store` 是抽象，引擎按依赖注入接入任意后端；元数据 filter 会**下推**到后端的原生查询，后端表达不了的部分在内存兜底。

| 后端 | 向量检索 | 元数据下推 | 全文(CONTAINS)下推 | 安装 |
|---|---|---|---|---|
| `LocalStore` | numpy FlatIndex（精确，列式过滤） | ✓ numpy 掩码 | 内存 | 内置 |
| `FaissStore` | faiss（精确） | ✗（残余内存过滤） | 内存 | `nlql[faiss]` |
| `HnswStore` | hnswlib（ANN，**sublinear**） | ✓ 过取+内存 | 内存 | `nlql[hnsw]` |
| `QdrantStore` | Qdrant（ANN） | ✓ 原生 Filter | 内存 | `nlql[qdrant]` |
| `ChromaStore` | Chroma（ANN） | ✓ 原生 where | 内存 | `nlql[chroma]` |
| `PgVectorStore` | Postgres+pgvector | ✓ SQL | ✓ **ILIKE** | `nlql[pgvector]` |

同一查询跨 Local/Faiss/Hnsw/Qdrant/Chroma **五后端结果逐条一致**（有测试保证）。

## 两段式检索：召回 + 重排

双塔向量召回对「短 chunk vs 长查询」只是粗匹配。挂一个 `Reranker` 做第二段：召回过取候选，再对每个 `(query, passage)` 联合精排。

```python
from nlql import Engine, CrossEncoderReranker   # 或自定义实现 Reranker 协议

engine = Engine(embedder, reranker=CrossEncoderReranker(), rerank_factor=5)
engine.search('SELECT SENTENCE LET rel = SIMILARITY(content, "...") ORDER BY rel DESC LIMIT 5')
# 也可按查询覆盖：engine.search(q, reranker=my_reranker, rerank_query="...")
```

见 [`examples/reranking.py`](examples/reranking.py)。

同一查询在三者上**结果逐条一致**（见 [`examples/hybrid_stores.py`](examples/hybrid_stores.py)）：

```python
from nlql.store.qdrant_store import QdrantStore
engine = nlql.Engine(embedder, store=QdrantStore(location=":memory:"))
```

## 多语言 · 类型 · 多模态

```python
from nlql import Engine, TypeTag
from nlql.embed import FakeMultimodalEmbedder
from nlql.model import Document, Modality, Payload

# 声明字段类型 → 带类型提示的比较（TEXT 抑制数值强转、DATE 按日期排序/比较）
engine = Engine(embedder, field_types={"published": TypeTag.DATE, "code": TypeTag.TEXT})

# 多模态：文本与图像同空间，文本 query 检索图像（IR/索引路径不变）
# 生产可用 DoubaoVisionEmbedder(托管,纯HTTP) 或本地 ClipEmbedder；下面用离线 Fake 演示
mm = Engine(FakeMultimodalEmbedder(), granularity="chunk")
mm.add_image(image_bytes_or_path_or_url, metadata={"kind": "photo"})   # 加多模态数据就这一行
mm.search('SELECT CHUNK LET rel = SIMILARITY(content, "a fluffy cat") ORDER BY rel DESC')
```

- **分词**：内置规则句子分割（中/英/日 + CJK 标点）；`LanguageRouter` 语言路由 + 可选 pysbd(缩写鲁棒，`nlql[segment]`)，可注册覆盖 `SENTENCE`。
- 示例：`examples/hybrid_stores.py`（跨后端一致）、`examples/multimodal_search.py`（图文互搜）。

## 状态

v2 重构进行中：**M0–M5 全部完成**（6 向量后端 + 下推、LLM 面、多语言/类型、多模态含真机 vision + 外部库对接 + **named 多向量** + PDF 抽图、**召回+重排两段式**、DOCX/PDF 加载）；M6 已 profiling + numpy 列式过滤（过滤查询 ~19×）+ HnswStore(sublinear)，Rust 暂缓。293 测试 / 90% 覆盖 / mypy strict / ruff 全过。
