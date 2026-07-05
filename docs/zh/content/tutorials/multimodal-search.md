# 多模态检索

多模态 embedder 把文字与图像映射到同一向量空间。文本查询检索图像时走与文本检索完全相同的查询路径，区别只在于向量的来源。本例用 `FakeMultimodalEmbedder` 离线演示。

```python
import nlql
from nlql.embed import FakeMultimodalEmbedder
from nlql.model import Modality, Payload

engine = nlql.Engine(FakeMultimodalEmbedder(), granularity="chunk")
```

`granularity="chunk"` 让写入的每个文档作为一个检索单元，适合以图像为单位的集合。

## 写入图像与文本

`Payload(Modality.IMAGE, data)` 描述一份图像载荷。把图像文档与普通文本文档一起 `add_documents`，引擎会按各自模态生成向量。

```python
engine.add_documents(
    [
        nlql.Document(
            id="cat",
            payloads=[Payload(Modality.IMAGE, b"a photo of a fluffy cat")],
            metadata={"kind": "image"},
        ),
        nlql.Document(
            id="car",
            payloads=[Payload(Modality.IMAGE, b"a red sports car on a road")],
            metadata={"kind": "image"},
        ),
        nlql.Document(
            id="dog",
            payloads=[Payload(Modality.IMAGE, b"a happy dog running in a park")],
            metadata={"kind": "image"},
        ),
        nlql.Document.from_text("An article about adopting kittens and cats.", id="article"),
    ]
)
```

也可以用 `engine.add_image(bytes_or_path_or_url, id=..., metadata=...)` 写入单张图像，参数接受原始字节、本地文件路径或 `http(s)` / `data:` URL。

## 用文本检索图像

查询语句不变。`SIMILARITY(content, "…")` 在多模态 embedder 上对图像单元同样有效。

```python
query = 'SELECT CHUNK LET rel = SIMILARITY(content, "fluffy cat kitten") ORDER BY rel DESC LIMIT 3'
print("text query 'fluffy cat kitten' retrieves across modalities:\n")

for unit in engine.search(query):
    modality = unit.payload.modality.value
    print(f"  [{unit.scores['rel']:+.3f}] ({unit.doc_id}, {modality})")
```

返回结果跨越图像与文本两种模态，按相关度排序。每个 `unit` 的 `payload.modality` 标明这条结果来自图像还是文本。

## 生产环境替换 embedder

`FakeMultimodalEmbedder` 仅用于演示与测试。生产中替换为真实的视觉 embedder：

```python
from nlql.embed import ClipEmbedder       # 或 DoubaoVisionEmbedder

engine = nlql.Engine(ClipEmbedder(model="openai/clip-vit-base-patch32"),
                     granularity="chunk")
```

- `ClipEmbedder` 基于 OpenAI CLIP，需安装 `pip install "python-nlql[local]"`。
- `DoubaoVisionEmbedder` 走豆包视觉模型 API，调用方式与其它 OpenAI 兼容 embedder 一致。

替换后查询与写入代码完全不变。

!!! tip "图像加入文本集合"
    把图像和文本写入同一个引擎，文本查询会同时命中两类内容。`payload.modality` 可用于在结果中区分来源。

## 下一步

- [快速开始](./quickstart.md)
- [两段式重排](./reranking.md)
- [Payload 与 Unit 数据模型](../concepts/overview.md)
- [Embedder API](../../reference/embed.md)

---

**完整源码**：[`examples/multimodal_search.py`](https://github.com/natural-language-query-language/python-nlql/blob/main/examples/multimodal_search.py)
