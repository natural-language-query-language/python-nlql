# 写入流程

一篇文档进入 NLQL 后，依次经过规整、切分、向量化、落索引四步。这些步骤都在写入时完成，查询阶段直接复用结果。

```python
import nlql

engine = nlql.Engine(nlql.OpenAIEmbedder(base_url="...", api_key="..."))
doc_id = engine.add_text(
    "AI agents plan tasks. They keep memory and call external tools.",
    metadata={"status": "published", "year": 2026},
)
results = engine.search(
    'SELECT SENTENCE LET rel = SIMILARITY(content, "how agents work") ORDER BY rel DESC LIMIT 3'
)
```

## 四个步骤

### 1. 规整（normalize）

文本进入前先做规整：统一空白与换行，去掉干扰分词的格式差异。这一步保证同一段文字无论以何种换行方式写入，都会切成同样的单元、命中同一份 embedding 缓存。

### 2. 切分（split）

规整后的文本按当前粒度对应的分词器切成单元。默认按句切（`SENTENCE`），内置分词器覆盖中、英、日及 CJK 标点。

```python
engine = nlql.Engine(nlql.FakeEmbedder(), granularity="sentence")  # 默认
engine = nlql.Engine(nlql.FakeEmbedder(), granularity="chunk")     # 改用 chunk 分词器
```

切分在写入时完成、查询时复用——`SELECT SENTENCE` 与 `SELECT SPAN(SENTENCE, window => n)` 返回的边界都来自这一步，不会出现查询时临时重切。

### 3. 向量化（embed）

每个单元的文本走 embedder 向量化一次，向量做单位归一化。引擎默认用 `CachedEmbedder` 包裹传入的 embedder，相同文本只计算一次（缓存细节见 [索引与缓存](./index-cache.md)）。

### 4. 落索引（index）

带向量的单元连同元数据写入存储。默认是进程内的 `LocalStore`，也可以是 Qdrant、Chroma、Faiss、HnswLib、pgvector——切换后端不改写入代码。

## 写入入口

引擎提供四种粒度的写入 API，内部都走同一条流水线：

```python
# 单段文本
engine.add_text("一段文字。第二句。", metadata={"status": "published"})

# 单个文件（按扩展名分派：.txt / .md / .docx / .pdf）
engine.add_file("notes.md")
engine.add_files(["a.txt", "b.md", "report.pdf"])

# 结构化文档（多 payload、多模态、自定义 id）
from nlql import Document, Payload
engine.add_documents([
    Document(id="d1", payloads=[Payload.text("...")], metadata={"kind": "note"}),
])
```

`add_file` 按扩展名选择加载器：`.txt` / `.md` 用内置加载器，`.docx` 需要 `python-nlql[loaders]`（python-docx），`.pdf` 需要同样的 extras（pypdf）。加载器把文件解析成 `Document`，再走同一条流水线。

```python
import nlql

engine = nlql.Engine(nlql.FakeEmbedder())
ids = engine.add_files(["agents.txt", "rag.md"])
print(f"loaded into {len(engine)} units: {ids}")
```

## 粒度选择

`granularity` 决定写入时切成什么样的单元，直接影响检索行为：

- **`sentence`**（默认）—— 每句一个单元，粒度细，相关度定位精确。适合需要精确指向某一句的问答、引用场景。
- **`chunk`** —— 按较大片段切分，每个单元信息更完整。适合答案需要完整上下文段落的 RAG。
- **自定义粒度** —— 注册自己的分词器即可（见 [注册与扩展](./registry.md)），比如按段落、按章节。

```python
engine = nlql.Engine(nlql.FakeEmbedder(), granularity="chunk")
engine.add_file("long_document.md")
# 每个 chunk 是一个检索单元
```

!!! note "粒度在写入时固定"
    引擎以一个 `granularity` 落库。查询时可以请求更大的返回单元（`SELECT SPAN(SENTENCE, window => 2)` 把命中句和前后句拼起来，或 `SELECT DOCUMENT` 取整篇），但不能再切到比写入粒度更细。

## 元数据

`metadata` 是用户自由使用的字段空间，写入时随文档挂上，查询时通过 `meta.<字段名>` 路径访问，与 `content` 同为字段。

```python
engine.add_text(
    "Retrieval-augmented generation grounds LLM answers in your documents.",
    metadata={"status": "published", "year": 2026, "topic": "rag"},
)
# 查询时：meta.status == "published" AND meta.year > 2024
```

!!! tip "声明字段类型让比较更准"
    构造引擎时传 `field_types={"year": TypeTag.DATE}` 等，能让日期按日期比、数值按数值比，避免字符串比较的歧义。

## 下一步

- 向量如何被缓存复用见 [索引与缓存](./index-cache.md)。
- 自定义分词器如何接入见 [注册与扩展](./registry.md)。
- 完整的写入与检索示例见 [快速开始](../tutorials/quickstart.md)。
- Engine 写入 API 的完整签名见 [SDK 参考](../../reference/sdk.md)。
