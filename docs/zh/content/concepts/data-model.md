# 数据模型

NLQL 的数据模型由四类对象组成：`Modality` 标识内容类型，`Payload` 承载具体内容，`Document` 是写入单位，`Unit` 是检索与返回的基本单位。模型从设计上与模态无关——文本只是 `Payload` 的一种。

```python
from nlql import Document, Payload, Unit, Modality
```

## Modality 与 Payload

`Modality` 是一个枚举，目前有 `TEXT`、`IMAGE`、`BLOB` 三种。`Payload` 把内容和它的模态绑在一起：

```python
from nlql import Payload, Modality

text_payload = Payload.text("一段文本")              # modality=TEXT, data=str
image_payload = Payload(modality=Modality.IMAGE,
                        data=image_bytes, mime="image/png")
```

`Payload.as_text` 在内容是文本时返回字符串，否则返回空串。这让文本与图像走同一套索引与查询路径，只是嵌入方式不同。

## Document

`Document` 是写入的单位，持有一到多个 `Payload` 加业务元数据：

```python
doc = Document.from_text(
    "AI agents plan tasks and call external tools.",
    id="doc-0",
    metadata={"status": "published", "topic": "agents"},
)
```

`Document.from_text` 是单个文本 payload 的便捷构造；多模态文档可以传多个 payload。`metadata` 是用户自由使用的字典，查询时通过 `meta.字段名` 访问。

```python
doc = Document(
    id="doc-1",
    payloads=[text_payload, image_payload],
    metadata={"author": "ada"},
)
```

## Unit

写入时，Document 被切分成多个 `Unit`。`Unit` 是检索与返回的原子单位，携带内容、向量、元数据与具名分数：

| 字段 | 含义 |
|---|---|
| `kind` | 粒度：`document` / `chunk` / `sentence` / `span` |
| `payload` | 该单元的内容 |
| `vector` | 写入时计算的 embedding |
| `metadata` | 从文档继承的业务元数据 |
| `scores` | 查询期间附加的具名分数，如 `{"relevance": 0.87}` |
| `span` | 当 `kind="span"` 时的上下文窗口信息 |
| `ordinal` | 该单元在文档中的位置序号 |

`Unit.content` 是文本内容的便捷访问；非文本 payload 返回空串。`scores` 让一条查询携带多个语义分数（例如同时算 `relevance` 与 `novelty`），按其中一个排序时另一个仍可读取。

## 粒度

`SELECT` 子句决定查询返回的粒度：

```sql
SELECT SENTENCE       -- 按句返回
SELECT SPAN(SENTENCE, window => 1)   -- 句子及其左右各 1 个邻居
SELECT CHUNK          -- 按写入时切分的块返回
SELECT DOCUMENT       -- 整篇文档
```

写入时用的切分器同时服务于查询时的粒度变换，因此 `SPAN` 能基于稳定的单元序号展开上下文窗口，而不需要在查询时重新切分文本。

## 写入入口

Engine 提供三个写入入口，覆盖从便捷到结构化的需求：

```python
engine.add_text("一段文本", id="doc-0", metadata={"status": "published"})
engine.add(Document.from_text("...", id="doc-1"))
engine.add_documents(iterable_of_documents, batch=256)
```

写入管线对每个 Unit 计算一次向量并按内容哈希缓存，重复内容不会重复嵌入。系统字段（`doc_id`、`kind`、`span`、`scores`）与业务 `metadata` 分开存放，不会相互污染。

## 多模态

`MultimodalEmbedder` 把文本与图像嵌入同一向量空间，因此可以用文字检索图像，查询语句与文本检索一致：

```python
mm = Engine(MultimodalEmbedder(), granularity="chunk")
mm.add_image(image_bytes, metadata={"kind": "photo"})
mm.search('SELECT CHUNK LET rel = SIMILARITY(content, "一只猫") ORDER BY rel DESC')
```

数据模型不区分文本与图像的检索路径——区别只在 embedder 如何向量化内容。

!!! note "命名向量"
    一个 Unit 可携带多组向量（`vectors` 字典），如同时有文本向量与图像向量。查询时用 `SIMILARITY(vec.<name>, "…")` 指定使用哪一组。

## 下一步

- [查询语法](./syntax.md)：如何在查询中引用 `content`、`meta.*` 与粒度
- [架构](./architecture.md)：写入管线的各阶段
- [API 参考 · model](../../reference/model.md)：`Document` / `Payload` / `Unit` 字段
- [多模态检索](../tutorials/multimodal-search.md)：文本检索图像的完整示例
