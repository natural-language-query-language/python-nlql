# 多模态

## 文本与图像在同一向量空间

多模态检索的前提，是把文本和图像映射到同一个向量空间。一旦两者同空间，一个文本查询的向量就能与一张图像的向量算相似度——文字可以直接检索图像。NLQL 的数据模型从设计起对模态无关，因此这条路径不引入新的查询语法：图像只是另一种 `Payload`，图像单元只是带图像向量的 `Unit`。

```python
import nlql
from nlql.embed import FakeMultimodalEmbedder

mm = nlql.Engine(FakeMultimodalEmbedder(), granularity="chunk")
mm.add_image(b"a photo of a fluffy cat", metadata={"kind": "photo"})
mm.add_text("An article about adopting kittens.", metadata={"kind": "text"})

for unit in mm.search('SELECT CHUNK LET rel = SIMILARITY(content, "fluffy kitten") ORDER BY rel DESC LIMIT 3'):
    print(f"  [{unit.scores['rel']:+.3f}] {unit.doc_id} ({unit.payload.modality.value})")
```

文字查询 "fluffy kitten" 同时命中图像与文本单元，按相关度排序。查询语句与纯文本检索完全一致。

## MultimodalEmbedder

`MultimodalEmbedder` 扩展了普通 `Embedder`，增加一个 `embed_images` 方法。关键是文本与图像共享同一个向量空间——`embed` 处理文本，`embed_images` 处理图像，两者产出的向量可以直接互相算 cosine。

```python
from typing import Protocol, runtime_checkable
import numpy as np
from nlql.embed.base import Embedder

@runtime_checkable
class MultimodalEmbedder(Embedder, Protocol):
    def embed_images(self, images: list[bytes | str]) -> np.ndarray:
        """返回 (n, dim) 的单位归一化图像向量矩阵。"""
        ...
```

`embed_images` 接收字节序列（或图像路径 / URL），返回与 `embed` 同维度、同空间的向量。引擎写入图像时调用它，写入文本时调用 `embed`；两条路径最终落到同一个索引里。

## 生产用的多模态后端

`FakeMultimodalEmbedder` 是离线、确定性的测试替身：它把图像字节当作文本描述嵌入，无需任何模型或网络。生产环境应换成真实的多模态模型：

- **`DoubaoVisionEmbedder`**——火山方舟托管的 `doubao-embedding-vision`（2048 维），纯 HTTP 调用，不依赖 torch，适合云端部署。
- **`ClipEmbedder`**——本地运行的 CLIP（需要 `nlql[local]`），文本与图像同空间，适合离线或私有部署。

切换只需把构造函数里的 embedder 换掉，查询代码、索引路径、过滤行为全部不变。

```python
from nlql.embed import DoubaoVisionEmbedder   # 或 ClipEmbedder
mm = nlql.Engine(DoubaoVisionEmbedder(), granularity="chunk")
```

## 写入图像

`engine.add_image` 是图像写入的便捷入口，对称于 `add_text`。它接收字节、路径或 URL，以及元数据：

```python
mm.add_image(image_bytes, metadata={"source": "s3://bucket/cat.jpg", "kind": "photo"})
mm.add_image("/data/photos/dog.png", metadata={"kind": "photo"})
```

!!! tip "存引用，不存字节"
    生产环境推荐把图像本体放在对象存储（S3、OSS），记录里只存向量、元数据与图像引用（URL 或 key）。`Payload(IMAGE, uri_str)` 或 `metadata["image_url"]` 都可以；记录保持轻量，向量库的检索性能不受影响。

## 跨模态检索的查询

查询语句与文本检索没有差别。`SIMILARITY(content, "文本")` 对图像单元同样适用——`content` 是模态无关的字段路径，引擎按单元的模态取对应向量参与打分。

```sql
SELECT CHUNK
LET   rel = SIMILARITY(content, "a red sports car")
WHERE  meta.kind == "photo"
ORDER BY rel DESC
LIMIT 5
```

元数据过滤照常生效，并且能交给支持原生过滤的后端（如 Qdrant、Chroma）在召回时就完成——这条性质对图像记录与文本记录同样成立。

## 与向量库结合

图像向量与文本向量本质都是向量，因此可以存进任何支持向量检索的后端。用一个多模态 embedder 把两者放进同一空间后，图像单元即存进 Qdrant / Chroma / PgVector，被文本查询跨模态检索。已验证 Chroma 下文本查询命中图像记录、且元数据过滤由 Chroma 原生完成。

## 下一步

- 多模态写入与检索的可运行示例：见仓库 `examples/multimodal_search.py`
- 一条记录挂多个向量分别可查：见 [多向量示例](https://github.com/natural-language-query-language/python-nlql/blob/main/examples/multivector.py)
- 后端如何处理元数据过滤：见 [Store 接口](./store-protocol.md)
- embedder 协议与缓存：见 [Embedder 参考页](../../reference/embed.md)
