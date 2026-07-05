# 注册与扩展

NLQL 的算子、函数、分词器、embedder 都通过同一套注册协议接入。注册之后就能在 NLQL 语句、Query Builder、LLM IR 三种入口里使用，无需修改文法。

## 统一注册

内置能力（`SIMILARITY`、`CONTAINS`、`LENGTH` 等）在导入 `nlql` 时自动注册。用户扩展走完全相同的路径：

```python
import nlql
from nlql import Signature, TypeTag

@nlql.register_function(
    "WORD_COUNT",
    signature=Signature((TypeTag.TEXT,), TypeTag.NUMBER),
)
def word_count(text: str) -> float:
    return float(len(text.split()))
```

注册后，`WORD_COUNT` 和内置函数一样可用：

```sql
SELECT SENTENCE
LET wc = WORD_COUNT(content)
WHERE wc > 10 AND meta.status == "published"
ORDER BY wc DESC LIMIT 5
```

```python
from nlql import QueryBuilder as Q, F
results = engine.search(Q.select("sentence").let("wc", Q.call("WORD_COUNT", Q.content())).where(F("wc") > 10))
```

注册时声明的 `Signature` 用于参数个数与类型检查。如果不关心类型，可以省略。

## 注册的位置

注册有两个层级，对应不同的可见范围。

**进程级注册**——所有引擎实例共享，适合工具函数、可复用的扩展：

```python
import nlql

@nlql.register_function("MY_FN")
def my_fn(text: str) -> float: ...
```

**实例级注册**——只对当前引擎生效，不泄漏到其它实例：

```python
engine = nlql.Engine(nlql.embed.FakeEmbedder())

@engine.register_function("TEMP_SCORE")
def temp_score(text: str) -> float: ...
```

实例级注册会覆盖同名的进程级注册——查找时先查实例，再查父级。这让同一个进程里运行多个引擎时，各自的扩展互不干扰。

## 自定义分词器

写入流程按 `granularity`（默认 `SENTENCE`）调用对应的分词器把文档切成单元。默认的句子分词覆盖中、英、日及 CJK 标点。需要更鲁棒的分词时（缩写、专业领域），可以注册自己的分词器：

```python
import nlql
import pysbd  # pip install python-nlql[segment]

@nlql.register_splitter("SENTENCE", overwrite=True)
def pysbd_sentences(text: str) -> list[str]:
    seg = pysbd.Segmenter(language="en", clean=False)
    return seg.segment(text)

engine = nlql.Engine(nlql.embed.FakeEmbedder())  # 写入时自动用上面的分词器
```

分词器在写入和查询时被同一套机制使用，因此 `SELECT SENTENCE` / `SELECT SPAN(SENTENCE, window => n)` 返回的边界与写入时一致，不会出现查询时临时重切导致的不匹配。

可以注册新的粒度名（比如 `"paragraph"`），并在构造引擎时指定：

```python
@nlql.register_splitter("PARAGRAPH")
def split_paragraphs(text: str) -> list[str]:
    return [p for p in text.split("\n\n") if p.strip()]

engine = nlql.Engine(nlql.embed.FakeEmbedder(), granularity="paragraph")
```

## 自定义 embedder

任何 embedder 都是一个实现了 `Embedder` 协议的对象。把它传给 `Engine` 即可——构造引擎只有这一种方式，没有按渠道区分的工厂方法。

```python
from nlql import Engine
from nlql.embed.base import BaseEmbedder
import numpy as np

class MyEmbedder(BaseEmbedder):
    @property
    def model_id(self) -> str:
        return "my-model"

    @property
    def dim(self) -> int:
        return 128

    def embed(self, texts):
        # 返回 (len(texts), dim) 的 float32 矩阵
        return np.random.rand(len(texts), self.dim).astype(np.float32)

engine = Engine(MyEmbedder())
```

OpenAI 兼容端点直接用 `OpenAIEmbedder(base_url=..., api_key=...)`，其它厂商写一个实现 `Embedder` 协议的类即可。多模态扩展只需额外实现 `embed_images`，把图像放进与文本同一向量空间。

## 能力声明

注册函数时可以附带能力声明，影响查询计划：

- `signature` — 参数与返回类型，用于类型检查和参数校验
- `pushdownable` — 提示该函数能被翻译成外部后端的原生查询（如 SQL 函数）
- `provides_score` — 标记该函数的值由召回阶段提供（`SIMILARITY` 即如此），不在每行调用

```python
@nlql.register_function(
    "LENGTH",
    signature=Signature((TypeTag.TEXT,), TypeTag.NUMBER),
    pushdownable=True,
)
def length(text: str) -> float:
    return float(len(text))
```

这些声明是可选的元信息。不声明时，函数作为普通标量函数逐行求值；声明后，规划器会据此决定哪些部分能交给后端、哪些留在内存。

## 下一步

- 写入流程如何使用注册的分词器见 [写入流程](./ingestion.md)。
- 自定义函数在查询中如何求值见 [执行流程](./evaluation.md)。
- 完整的注册 API 见 [Registry 参考](../../reference/registry.md)。
- 自定义 embedder 的协议定义见 [Embedder 参考](../../reference/embed.md)。
