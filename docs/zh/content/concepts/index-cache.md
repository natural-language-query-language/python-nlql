# 索引与缓存

向量在写入时计算并落入索引，查询时不再重复计算。这是 NLQL 查询延迟较低的根本原因，也决定了相似度分数的取值范围。

```python
import nlql

engine = nlql.Engine(nlql.embed.OpenAIEmbedder(base_url="...", api_key="..."))
engine.add_text("AI agents plan tasks and call external tools.")
engine.add_text("Banana bread is a quick loaf made with ripe bananas.")

results = engine.search(
    'SELECT SENTENCE LET rel = SIMILARITY(content, "artificial intelligence") ORDER BY rel DESC'
)
# rel 落在 [-1, 1] 区间；相关文档约 0.3+，无关文档接近 0 或为负
```

## 写入时计算，查询时复用

写入一篇文档时，引擎把它切成单元（默认按句），对每个单元的文本做一次向量化，归一化后写入索引。查询阶段只对查询文本向量化一次，然后用一次矩阵乘法得到所有候选的相关度——不会对每条记录重新调用 embedder。

```python
engine.add_text("第一句。第二句。第三句。")
# 内部流程：normalize → split → embed → index
# 三个单元各自 embed 一次，向量进入索引
```

后续查询命中这些单元时，向量已在索引里，不需要任何向量化调用。只有查询文本本身会被 embed。

## EmbeddingCache

同一篇文档被重复写入、或不同文档包含相同文本片段时，重复向量化是浪费。`EmbeddingCache` 用内容寻址的方式避免这一点。

```python
from nlql import EmbeddingCache, CachedEmbedder, OpenAIEmbedder

cache = EmbeddingCache()
embedder = CachedEmbedder(OpenAIEmbedder(base_url="...", api_key="..."), cache)

engine = nlql.Engine(embedder)
engine.add_text("Important fact repeated across documents.")
engine.add_text("Important fact repeated across documents.")  # 第二次零向量调用
```

缓存键由 `model_id + dim + modality + 文本内容` 的哈希组成。换模型或调整输出维度时，键自动失效，不会返回尺寸不匹配的旧向量。

!!! note "Engine 默认就带缓存"
    把任意 embedder 传给 `Engine` 时，如果不是 `CachedEmbedder`，引擎会自动包一层。所以多数情况下不用手动构造 `CachedEmbedder`，除非你想显式控制缓存实例。

### 持久化

缓存可以保存为 `.npz` 文件，进程重启后加载复用，避免冷启动时对同一批文本重新付费调用 API。

```python
cache.save("embeddings.npz")          # 落盘
cache.load("embeddings.npz")           # 读回
```

!!! tip "缓存键与文本归一化"
    缓存键基于已归一化的文本。多余的空白、不一致的换行不会产生不同的向量——同一文本无论怎么换行写法，命中同一份缓存。

## 分数是原始余弦值

`SIMILARITY` 返回的是两个归一化向量的点积，即原始余弦值，取值范围 `[-1, 1]`。

```python
for unit in engine.search(
    'SELECT SENTENCE LET rel = SIMILARITY(content, "machine learning") ORDER BY rel DESC'
):
    print(unit.scores["rel"], unit.content)
```

这是有意为之。一些系统会把余弦折算成 `(cos + 1) / 2` 让它落在 `[0, 1]`，但这会让阈值变得不直观：正交的两个向量本应是 0，折算后却成了 0.5。NLQL 直接用原始余弦：

- 强相关约 `0.3 ~ 0.6`
- 弱相关约 `0.1 ~ 0.2`
- 无关接近 `0` 或为负

写 `WHERE rel >= 0.3` 时，阈值与实际的几何关系对应得上，无论是开发者读查询，还是 LLM 通过 function-calling 产出 IR，都能据此设置合理的过滤条件。

## 索引后端

默认的 `LocalStore` 用纯 numpy 做精确最近邻——一次矩阵乘法完成全部候选打分，零原生依赖，适合中小数据量。数据规模上来后可以换近似最近邻后端，查询代码不变：

```python
from nlql.store.hnsw_store import HnswStore
from nlql.store.faiss_store import FaissStore

engine = nlql.Engine(embedder, store=HnswStore())      # hnswlib，亚线性召回
engine = nlql.Engine(embedder, store=FaissStore())      # Faiss
```

所有后端实现同一套 `Store` 接口，写入路径相同（向量化 → 归一化 → 落库），查询路径也相同。切换后端只影响性能特征，不影响结果。

## 下一步

- 查询阶段如何使用这些分数见 [执行流程](./evaluation.md)。
- 文档如何被切成单元见 [写入流程](./ingestion.md)。
- 多后端切换的实践见 [混合后端](../tutorials/hybrid-stores.md)。
- 各后端的性能对比见 [性能](../../performance.md)。
