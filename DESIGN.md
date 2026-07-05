# NLQL v2 — 商业级系统设计

> **一句话定位**：NLQL 是一个**面向 Agent / RAG 的语义检索中间件**——用一套 SQL 风格、语义清晰的查询语言（及其等价的结构化 IR），在纯文本、自持索引与外部向量库之上提供统一、可解释、高性能的检索能力。

本文档是 v2 重构的奠基设计。它基于对 v1（PoC）实现的评审，修正其在**性能、语义一致性、注册模型、类型系统、集成友好性、多模态**上的根本缺陷。

---

## 实现状态（2026-07）

**M0 骨架 · M1 文本 MVP · M2 混合引擎(Store 适配+下推) · M4 分词/类型 · M5 多模态 均已落地并全程测试**（Python 3.11+，本机验证于 3.14）。

- 模块：`model / errors / types / registry / ir / lang / embed / ingest / store / plan / exec / sdk` 全部实现。
- 测试：**293 用例、90% 覆盖率**（`pytest --cov`），mypy strict 无告警、ruff 全过；语义链路用确定性 `FakeEmbedder`/`FakeMultimodalEmbedder` 离线回归。
- 真实验证：`Engine(OpenAIEmbedder(base_url=…))` 已在真实 OpenAI 兼容端点（text-embedding-3-small, 512 维）跑通语义检索 + 元数据过滤，命名分数以诚实原始 cosine 呈现（相关 0.34 / 无关 -0.02）。
- 三入口等价：NLQL 字符串、Query Builder、LLM IR(JSON) 编译到同一 IR，结果逐条一致（有测试保证）。
- **M2 已验证**：`LocalStore`(numpy) / `FaissStore` / `QdrantStore` / `ChromaStore` 四后端同一查询**逐条结果一致、分数相同**；EXPLAIN 显示各自下推决策（Local/Qdrant/Chroma 推元数据、Faiss 全残余）。`PgVectorStore` 加 **CONTAINS→ILIKE 全文下推**（翻译单测；实况需 Postgres）。`text_pushdown` 能力位区分「能下推子串谓词」的后端。
- **M4 已验证**：中/英/日 sentence 分割；语言路由 `LanguageRouter` + 可选 pysbd(缩写鲁棒)；声明式 meta 字段类型驱动带提示比较（TEXT 抑制数值强转、DATE 按日期排序），类型字段留内存求值不下推。
- **M5 已验证**：`Payload{modality}` 图像入 ingestion；`MultimodalEmbedder` 协议（`embed`+`embed_images` 同空间）。离线 `FakeMultimodalEmbedder` 跑通文本→图像；**真机** `DoubaoVisionEmbedder`(火山方舟 doubao-embedding-vision, 2048维, 纯 HTTP 无 torch) 跑通**真实像素图文互搜**（"a dog"/"a bus"/"players" 各命中对应图）；可选 `ClipEmbedder`(本地 CLIP)。IR/索引路径不变，仅向量来源不同。
- **M6 已 profiling + Python 层优化落地**：语义 top-k 本就快(1.6ms/15k units)；瓶颈是带过滤查询的**逐行 Python 强转循环**（非向量数学）。两处优化：① `as_number/as_date` 前置判断；② `store/columns.py` **numpy 列式过滤**（强转按 distinct 值只算一次、掩码 numpy 组合，语义仍走 `compare_values` 故与逐行/跨库完全一致，有 `column mask == matches_filter` 测试）。过滤查询 **83→4.4ms(~19×)**，与语义 top-k 同量级。结论：**Rust 暂不值得**。`HnswStore`(hnswlib, sublinear) 已落地作百万级召回后端（纳入五后端一致性）。见 `benchmarks/README.md`。
- **两段式检索（召回+重排）**：可插拔 `Reranker` 协议（`rerank(query, units)`）+ `FakeReranker`(确定性) + `CrossEncoderReranker`(可选,联合打分 query+passage,治双塔长度不对称)；`Engine(reranker=…, rerank_factor=5)` / `search(reranker=…, rerank_query=…)`，召回过取→精排→limit，EXPLAIN 展示。**多模态入口** `engine.add_image(bytes|path|url, metadata=…)` 对称 `add_text`。

落地过程中对本设计做了 5 项精化，均已并入下文对应章节，并在此摘要：

1. **默认 ANN = 纯 numpy `FlatIndex`**（精确、零原生依赖，3.14 上稳装）；`hnswlib/faiss` 降为同协议可选后端。见 §7。
2. **算子并入函数**：`CONTAINS/MATCH/LIKE` 是返回 BOOL 的内置**函数**，中缀仅解析器语法糖 → `Call` 节点。文法只固定结构关键字 + 极小闭合中缀集。见 §4.3。
3. **Registry 作用域用父子链实现**（全局为根、实例为 child、查找 self→parent），比 `scope` 字符串更干净。见 §8。
4. **SIMILARITY = provider 型函数**（`provides_score=True`），值由召回阶段填入 `Unit.scores`，求值器仅在此一处按能力标志分派——「无特判」的原则性落地。见 §6。
5. **Engine 纯依赖注入**：`Engine(embedder)` 是唯一构造方式，不设 `with_openai/with_fake` 之类会随渠道膨胀的工厂——embedder 本身就是扩展点。所有 OpenAI 兼容渠道 = 一个 `OpenAIEmbedder(base_url=…)`；其它厂商 = 一个新的 `Embedder` 实现。契合「底层可扩展、便利上层封装」。见 §5、§11。

---

## 0. 设计原则

- **KISS / 正交**：字段访问、标量函数、谓词、逻辑组合语义正交，求值走单一路径，杜绝 special-case。
- **IR 优先**：查询的规范形态是 `Query IR`（可 JSON 序列化）。NLQL 字符串、Python Query Builder、LLM function-calling 都编译到同一个 IR。
- **索引是一等公民**：向量在写入时计算并落索引，查询走"召回 → 精排"，绝不全量重算。
- **注册即可用**：一切扩展（算子/函数/分词/embedder/模态）走统一注册协议；内置能力与用户能力同一路径，无占位死代码。
- **模态无关**：数据模型从第一天起对文本/图像/二进制通用；v1 仅落地文本实现。
- **可解释**：每个查询都能 `EXPLAIN`，产出查询计划、下推情况、预估代价——这是给 LLM 和开发者的信任基础。

---

## 1. v1（PoC）核心缺陷回顾

| 领域 | 缺陷 | 证据（v1） |
|---|---|---|
| 性能 | 每次查询对全部 unit 重算 embedding，纯 Python 点积，无索引/缓存 | `text/similarity.py:84`、`config.py:23 enable_caching=False`、`engine/routing.py:56 return RoutingPlan()` |
| 语义 | `SIMILAR_TO` 是假算子，全局仅一个分数，无法表达"既像 A 又像 B" | `text/similarity.py:143 return first`、`engine/evaluator.py:177` |
| 语义 | `META` 时而函数时而字段；`CONTAINS/MATCH` 1参/2参靠特判 | `parser/grammar.lark:41`、`engine/evaluator.py:158,168` |
| 注册 | 词法写死算子表 + 运行时注册两套不同步；内置算子是 placeholder 死代码 | `grammar.lark:41,45`、`registry/operators.py:22` |
| 类型 | 整套 `BaseType`/`wrap_value` 从未被比较逻辑调用（死代码），日期按字符串比 | `types/core.py`、`engine/evaluator.py:226` |
| 集成 | `add_*`/分块只在 `MemoryAdapter`，非统一抽象 | `adapters/memory.py:113` |
| 多模态 | 全系统 `content: str`，无入口 | `text/units.py:17`、`text/embedding.py:39` |
| 分词 | 仅标点正则，无多语言鲁棒 | `text/splitting.py:34` |

---

## 2. 整体架构

```
┌────────────────────────── Integration / SDK 层 ──────────────────────────┐
│  Python SDK  ·  Query Builder  ·  NLQL 字符串  ·  LLM function-calling schema │
└───────────────────────────────────┬──────────────────────────────────────┘
                                     │  编译到
                                     ▼
                            ┌─────────────────┐
                            │    Query IR     │  ← 规范形态、可 JSON 序列化、可 EXPLAIN
                            └────────┬────────┘
        写入路径                      │  查询路径
┌───────────────────────┐   ┌────────▼─────────────────────────────────────┐
│ Ingestion Pipeline    │   │ Planner  →  Executor                          │
│  Normalizer           │   │  · 拆分 下推(pushdown) vs 内存(in-memory)      │
│  Splitter(可插拔分词)  │   │  · 向量召回(ANN top-k) → 谓词过滤 → 精排/重排   │
│  Embedder(带缓存)      │   │  · 粒度变换(SENTENCE/SPAN)                     │
│  Indexer              │   └────────┬─────────────────────────────────────┘
└──────────┬────────────┘            │
           ▼                         ▼
┌──────────────────────────── Store 层 (统一接口) ──────────────────────────┐
│  LocalStore(自持: ANN 索引 + 元数据列 + 倒排 + 向量/结果缓存)                │
│  ExternalStore Adapters(Qdrant / Chroma / FAISS / PGVector …，支持下推)     │
└───────────────────────────────────────────────────────────────────────────┘
                                     ▲
                            ┌────────┴────────┐
                            │  Registry (统一) │  operator/function/splitter/embedder/modality
                            └─────────────────┘
```

**分层职责**
- **Integration/SDK**：面向使用者的入口，全部编译到 Query IR。
- **Query IR**：AST 之上的规范中间表示；序列化即 function-calling 载体。
- **Ingestion**：写入时完成 normalize → split → embed(cache) → index。
- **Planner/Executor**：把 IR 拆成"能下推的"与"必须内存算的"，走索引召回而非全量。
- **Store**：数据与索引的持有者；`LocalStore` 自持，`ExternalStore` 转译下推。
- **Registry**：所有扩展点的单一注册中心。

---

## 3. 核心数据模型（模态无关）

```python
# 示意，非最终签名
class Modality(str, Enum):
    TEXT = "text"; IMAGE = "image"; BLOB = "blob"

@dataclass
class Payload:
    modality: Modality
    data: str | bytes            # text=str, image/blob=bytes 或 URI
    mime: str | None = None

@dataclass
class Document:
    id: str
    payloads: list[Payload]      # 一个文档可含多模态
    metadata: dict[str, Any]     # 业务元数据（用户空间）
    source: str | None = None

@dataclass
class Unit:                      # 检索/返回的基本单位（chunk/sentence/span 的统一体）
    id: str
    doc_id: str
    kind: Literal["document","chunk","sentence","span"]
    payload: Payload
    vector: Vector | None        # 写入时计算
    span: SpanInfo | None        # kind=span 时的上下文窗口
    metadata: dict[str, Any]
    scores: dict[str, float]     # 具名分数：{"relevance": 0.87, ...}
```

要点：
- `content: str` → `Payload{modality, data}`：一步到位支持多模态；文本是其中一种。
- 相似度分数从"塞进 metadata 的魔法字段"改为**具名 `scores` 字典**，支持一条查询多个语义分数。
- 系统字段（doc_id/kind/span）与业务 `metadata` 分离，杜绝 v1 里系统字段污染 metadata 的问题。

---

## 4. 语法 v2（NLQL/2）——从"特判"到"正交"

### 4.1 语义模型（符号可在 M0 微调，语义是核心）

四类正交构件：
1. **路径（Path）**：`content`、`meta.status`、`meta.created_at`。元数据统一为**点路径访问**，不再有 `META("x")` 函数形态。
2. **标量函数（Scalar fn）**：`SIMILARITY(content, "…")`、`LENGTH(content)`、`COUNT(content, "…")`、自定义函数。统一 `NAME(args)`，返回值。
3. **谓词（Predicate）**：中缀 `content CONTAINS "x"`、`content MATCH /regex/`、`content LIKE "%x%"`，或返回 bool 的函数。主语显式，无 1参/2参歧义。
4. **逻辑组合**：`AND / OR / NOT`，括号分组。

**命名分数**：任意标量表达式可 `AS name` 绑定，别名在 `WHERE`/`ORDER BY` 复用——这让"多语义查询 + 按某个分数排序"成为一等能力。

### 4.2 对比

**v1（问题版）**
```sql
SELECT SPAN(SENTENCE, window=1)
WHERE SIMILAR_TO("AI Agent architecture") > 0.8   -- 假算子, 全局仅一个
  AND META("status") != 'draft'                    -- 函数形态的字段访问
  AND (MATCH("planning") OR MATCH("memory"))       -- 主语 content 靠猜
ORDER BY SIMILARITY DESC                            -- 魔法关键字
LIMIT 5
```

**v2（正交版）**
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
- `LET` 显式声明具名分数，可多个、可组合排序 → 根治"只能一个 SIMILAR_TO"。
- `meta.status` 与 `content` 同为路径，语义统一。
- `content CONTAINS "x"` 主语显式，无特判。
- 排序引用命名分数，无魔法关键字。

### 4.3 词法/语法工程
- 关键字与结构（`SELECT/WHERE/LET/ORDER BY/AND/OR/NOT`、中缀谓词、点路径）走文法；关键字大写、大小写敏感，避免与小写 metadata 键（`meta.status`、`meta.window`）冲突。
- **算子并入函数**：`CONTAINS/MATCH/LIKE` 是返回 BOOL 的内置**函数**，中缀（`content CONTAINS "x"`）只是解析器语法糖 → `Call("CONTAINS", [content, "x"])`。文法只固定「结构关键字 + 这一小撮闭合中缀谓词 + 比较符」。
- **函数名不写死进文法**：调用一律 `IDENT(args)`，合法性在语义分析阶段查 Registry。新增 `SIMILARITY/LENGTH/自定义` 函数**无需改文法**，根治"Token 表混乱"。
- 标识符统一按 `Path` 解析，解析后用 LET 别名集把 `Path(alias)`→`Ref`（`relevance`→Ref，`content`/`meta.status`→Path），边界清晰无歧义。

---

## 5. Query IR 与三入口

IR 是规范形态，三种入口等价编译到它：

```python
# 入口 A：NLQL 字符串
engine.search('SELECT SENTENCE WHERE ... LIMIT 5')

# 入口 B：Query Builder（SDK，类型安全）
Query.select("sentence")
     .let("relevance", Similarity("content", "AI agents"))
     .where(F("relevance") >= 0.8, Meta("status") != "draft")
     .order_by("relevance", desc=True).limit(5)

# 入口 C：结构化 JSON（LLM function-calling，最可靠）
{"select":{"unit":"sentence"},
 "let":{"relevance":{"fn":"SIMILARITY","args":["content","AI agents"]}},
 "where":{"op":">=","left":{"ref":"relevance"},"right":0.8},
 "limit":5}
```

- **LLM 集成核心**：暴露 IR 的 JSON Schema 作为 function-calling 定义；LLM 直接产出结构化 IR（比生成字符串更少语法错误），或产出字符串再由 parser 校验回填。
- `engine.explain(query)` 返回：解析后的 IR、Planner 的下推/内存拆分、召回策略、预估代价——供 Agent 自检与开发者调试。

---

## 6. 求值模型（统一表达式代数）

- 每个 IR 表达式节点求值为一个值（标量/bool/向量分数引用）；比较与逻辑节点消费下层值。
- **无 special-case**：`SIMILARITY`、`meta.*`、`CONTAINS`、自定义函数都是普通节点，区别只在"能否下推"与"求值代价"，由 Planner 决定，不由 evaluator 用 if-else 区分。
- 短路求值 + 代价排序：先算便宜的谓词（元数据/字符串），语义分数由索引召回阶段提供，避免逐条 embed。

**provider 型函数机制（如何做到「无特判」而 SIMILARITY 仍走索引）**：
- 注册表里 `SIMILARITY` 的 capability 标 `provides_score=True`，本身无 row-wise `impl`。
- Planner 遍历 IR 收集所有打分 `Call`，以 `score_key = canonical(call)`（`call.to_dict()` 的稳定 JSON）去重；LET 别名穿透到其绑定表达式，故 `LET relevance = SIMILARITY(content,"x")` 与内联同一调用共享一个 key、只算一次。
- Executor 在召回阶段把候选向量堆成矩阵，与「查询向量矩阵」一次 `matmul` 得到 cosine，写入 `unit.scores[score_key]`。
- Evaluator 求值 `Call` 时，**唯一的分支**是 `cap.provides_score`：为真则读 `unit.scores[score_key(call)]`，为假则 `cap.impl(*args)`。这是对能力元数据的分派，而非对函数名的硬编码——v1 的 `if node.operator=="SIMILAR_TO"` 被彻底消除。
- 类型接线：比较前做数值/日期规约（`meta.year > 2023` 数值比、ISO 日期按 datetime 比），null 参与有序比较则该行落选（SQL 式），修复 v1「日期按字符串比 + 类型系统死代码」。

---

## 7. 索引与缓存层（回应 v1 头号缺陷）

**写入时（Ingestion）**
- 每个 `Unit` embed 一次并**单位归一化** → 落 `LocalStore` 索引。**M1 默认索引 = 纯 numpy `FlatIndex`**：精确、零原生依赖、Python 3.14 稳装；`hnswlib/faiss` 作同 `Store` 协议的可选加速后端（`nlql[hnsw]`/`nlql[faiss]`）。
- Embedding 缓存：`sha256(model_id + dim + modality + normalized_content)` → vector（`dim` 入 key，换模型/截维自动失效），可持久化 `.npz`，杜绝重复计算——直击 v1 头号缺陷。
- 元数据列（`Unit.metadata`）支撑非语义谓词过滤；倒排为后续增强预留。

**查询时（Executor）**
- 归一化向量 ⇒ `matrix @ query` 即 cosine。M1 走「候选打分（一次 matmul，**绝不重算 embedding**）→ 谓词过滤 → 排序 → 粒度变换 → LIMIT」。`FlatIndex` 为 O(N) 向量化点积（换 hnswlib 即 sublinear，接口不变）。
- 无语义分数（纯元数据）→ 直接扫描过滤；外部库下推留待 M2。
- 结果缓存：`fingerprint(IR + store 版本)` → results（预留，M1 未启用）。

**相似度语义修正（已验证）**：v1 用 `(cos+1)/2` 导致阈值不直观（正交=0.5）。v2 直接用**原始 cosine ∈ [-1,1]**：真实 OpenAI 端点上「AI/神经网络」查询对 ML 句得 0.34、对香蕉面包得 -0.02，度量透明、阈值对 LLM/开发者可解释。

---

## 8. 统一注册协议（回应"注册系统混乱"）

单一 `Registry`，四类 capability 同构注册（`function` 已吸收 v1 的 operator——见 §4.3）：

```python
registry.register(
    kind="function",            # function | splitter | embedder | modality
    name="WORD_COUNT",
    impl=word_count,
    signature=Signature((TypeTag.TEXT,), TypeTag.NUMBER),  # 供 arity/类型检查、下推分析
    provides_score=False,       # True 表示值由召回阶段提供（如 SIMILARITY）
    pushdownable=False,         # M2 外部库下推提示
)
```

**作用域用父子链实现**，而非 `scope` 字符串：进程级 `GLOBAL_REGISTRY` 为根，每个 `Engine` 持一个 `child()`；查找 self→parent，实例注册天然 shadow 全局，`instance > global` 自然成立、且互不泄漏。装饰器糖：`@registry.function("NAME", ...)`。

- 命名、作用域、优先级、冲突策略**统一一处**定义。
- 内置能力与用户扩展走**同一注册路径**，删除 v1 的 placeholder 死代码。
- Parser/Planner 从 Registry 取"可调用名 + 签名 + 可下推性"，实现"注册即可用 + 可参与下推决策"。

---

## 9. Ingestion Pipeline（统一、数据源无关）

顶层引擎提供统一写入 API（不再散落在具体 adapter）：
```python
engine.add(Document(...))                    # 结构化文档
engine.add_text("…", metadata={...})         # 便捷入口
engine.add_documents(iterable, batch=256)    # 批量
```
内部：`Normalizer → Splitter → Embedder(cache) → Indexer`。
- **Splitter 可插拔**：默认规则分词，可注册 `pysbd`/`spaCy`/`jieba`/`nltk` 按语言路由，解决多语言鲁棒性。
- Splitter 也用于 SENTENCE/SPAN 粒度，写入与查询共用一套，避免 v1 查询时临时重切。

---

## 10. Store 适配与「下推」的澄清

**Store 抽象才是第一位的**——这正是把 Qdrant / Chroma / PgVector 直接对接进来的方式：每个 Store 用**自己后端的原生能力**实现 `ann_search + 过滤 + limit`，`StoreCaps` 声明它支持什么。

- `QdrantStore.ann_search(vector, k, filter)` 把 filter 翻译成 Qdrant 原生 filter，让 Qdrant 只返回命中向量；`ChromaStore`→`where`；`PgVectorStore`→SQL `WHERE` + `<=>`。「把查询翻译成后端原生查询」这一步**就是下推**——它是每个适配器的**内部细节**，不是一个独立的大子系统。
- **「下推」并非总是必要**：若后端能表达整条查询，就是纯委托、无拆分。只有当查询用到后端表达不了的能力（自定义 Python 函数谓词、复杂正则、跨源逻辑）时，才把「可下推子集」交给后端、**残余部分**在返回的候选上做内存后置过滤。这个「拆分」是回退路径，不是常态。
- 因此接口约定：传给 Store 的 filter 应是**数据（IR 子集）而非编译好的 Python 闭包**，这样每个适配器都能自主翻译。`LocalStore` 把该 IR 编成 numpy 掩码，外部 Store 翻成原生 filter。（M1 的过滤在 Executor 内完成；M2 落地「IR filter 传入 Store」接口。）
- `LocalStore` 天生全能力；外部 Store 尽力下推、回退内存。补齐 v1 `routing.py` 空壳。

> 一句话：你说的「抽象 Store 层、让实现直接对接 Qdrant/Chroma/PgSQL」**就是我们要做的**；"下推" 只是「适配器把查询翻译成后端原生查询」的名字 + 后端不支持时的内存兜底，不是额外负担。

---

## 11. Store / Adapter 接口

```python
class Store(Protocol):
    def upsert(self, units: list[Unit]) -> None: ...
    def ann_search(self, vector, k, filter=None) -> list[Unit]: ...   # 支持下推 filter
    def scan(self, filter=None) -> Iterable[Unit]: ...
    def capabilities(self) -> StoreCaps: ...                          # 声明可下推能力
```
- `LocalStore`：`hnswlib`/`faiss` + 元数据字典/倒排，进程内或本地持久化。
- `ExternalStore`：Qdrant/Chroma/PGVector；`capabilities()` 驱动 Planner 下推。
- 写入 API 统一在引擎层，adapter 只实现存取原语——修复 v1 "add 只在 Memory" 的问题。

---

## 12. 类型系统（真正接线）

- 字段类型来自：注册的 `meta` 字段类型 + 从数据推断 + 函数 `Signature.returns`。
- 比较在 IR 求值/下推前做**类型解析与规约**（日期按日期比、数值按数值比），修复 v1 "日期按字符串比" 的 bug。
- 类型信息同时服务于下推（外部库需要正确类型的 filter）。

---

## 13. 多模态与文档加载（已落地）

**多模态（M5 完成）**：`Payload{modality}` 模态无关；`MultimodalEmbedder` 协议（`embed` + `embed_images` 同空间）。已落地 `FakeMultimodalEmbedder`(离线)、`DoubaoVisionEmbedder`(火山方舟托管, 真机图文互搜)、`ClipEmbedder`(本地可选)。查询侧 `SIMILARITY(content, "文本")` 对图像 Unit 天然适配——同一 IR、同一索引路径，只是 Unit 向量来自图像。入口 `engine.add_image(bytes|path|url, metadata=…)`。

**对接多模态向量库（关键澄清）**：**Store 模态无关**——图像向量与文本向量一样是向量。用一个多模态 embedder(CLIP/doubao-vision)把两者放进**同一空间**，图像 Unit 的向量即可存进 Qdrant/Chroma/PgVector，被文本 query 跨模态检索。已用 `ChromaStore` 验证：文本 query 命中图像记录，且**元数据过滤下推到 Chroma 对图像记录同样生效**。
- **生产建议**：外部库存**向量 + 元数据 + 图像引用(URL/S3 key)**，而非图像 bytes——`Payload(IMAGE, uri_str)` 或把引用放 `meta.image_url`，bytes 留对象存储，records 轻量。
- **named 多向量（已落地）**：`Unit.vectors: dict[name→Vector]` + `Unit.get_vector(name)`；`SIMILARITY(vec.<name>, "…")` 按名打分（`content`→默认向量）。`Engine(named_embedders={"image": …})` 配每名 embedder，`engine.add_multivector(id, content=…, named={"image": bytes, "title": str})` 一条记录挂多向量分别可查。Executor：命名 scorer 走 scan + `unit.get_vector(name)` 逐 scorer matmul（各向量空间独立）；外部库里命名向量随 Unit 在内存中可查，**native named-vector 索引**（Qdrant named vectors）留作外部适配增强。

**文档加载（`nlql.loaders`）**：`Loader` 协议 + 按扩展名分派（可 `register_loader` 扩展）；内置 `TextLoader`(.txt/.md) / `DocxLoader`(python-docx) / `PdfLoader`(pypdf, 可逐页, `extract_images=True` 抽图为 image payload)。`engine.add_file(path)` / `add_files(paths)` 加载为 `Document` 后走同一写入管线（文本切句、图像成图单元）。`nlql[loaders]` 可选依赖、懒导入。

---

## 14. Rust 加速路线（预留，不前置）

- v1 全 Python + `numpy`/`hnswlib`/`faiss`，正确性与 API 优先。
- 热点候选：批量向量距离、分词、IR 求值内核。经 profiling 确认后用 `PyO3`/`maturin` 下沉，接口层保持不变（可替换实现）。
- 不为未验证的性能假设提前写 Rust。

---

## 15. 目录结构（建议）

```
src/nlql/
  model/        # Payload, Document, Unit, Vector, Span
  registry/     # 单一 Registry + capability 定义
  lang/         # grammar, lexer, parser → IR
  ir/           # Query IR 节点、JSON 序列化、schema 导出
  ingest/       # normalizer, splitter(可插拔), embedder(cache), indexer
  plan/         # Planner, 下推分析, cost
  exec/         # Executor, 求值内核, 重排
  store/        # Store 协议, LocalStore, external/*(qdrant,chroma…)
  embed/        # Embedder 协议 + FakeEmbedder/OpenAIEmbedder/(SentenceTransformers 可选) + 缓存
  sdk/          # Engine, Query Builder, EXPLAIN, function-calling schema
  types/        # 类型系统
  errors.py
tests/  examples/  docs/  benchmarks/
```

---

## 16. 实施路线图（里程碑 + 验收）

| 里程碑 | 内容 | 验收标准 | 状态 |
|---|---|---|---|
| **M0 骨架** | 目录结构、`model/`、单一 `Registry`、`ir/` 节点与 JSON、`lang/` parser→IR、`sdk/Engine` | NLQL 字符串与 Query Builder 都能编译到同一 IR，round-trip 测试通过 | ✅ 完成 |
| **M1 文本 MVP** | `LocalStore`(numpy FlatIndex) + Embedder 缓存 + Ingestion + Executor 全链路 | 语义检索不重算 embedding；替代原 PoC 全部能力，测试全绿 | ✅ 完成 |
| **M2 Store 适配** | `Store` 适配器直连 Faiss/Qdrant/Chroma/PgVector + IR filter 下推(含 CONTAINS→ILIKE 全文) + `BaseUnitStore` 复用 | 同一查询跨 Local/Faiss/Qdrant/Chroma 逐条一致，翻译/残余在 EXPLAIN 可见 | ✅ 完成 |
| **M3 LLM 集成面** | function-calling schema、`EXPLAIN`、Query Builder 完备 | 给定 schema，LLM 产出的 IR 可直接执行；EXPLAIN 输出计划 | ✅ 面已就绪（schema/tool/EXPLAIN/Builder + `search_ir`） |
| **M4 鲁棒分词/类型** | 可插拔多语言分词(语言路由+pysbd)、声明式字段类型带提示比较 | 中/英/日分词与日期/数值/TEXT 比较用例通过 | ✅ 完成 |
| **M5 多模态** | Image payload 入 ingestion + Multimodal/CLIP embedder(同空间) | 文本→图像互搜跑通，IR/索引路径不变 | ✅ 完成 |
| **M6 Rust 加速** | 热点 profiling → PyO3 下沉 | 关键路径基准提升，API 不变，测试全绿 | 🟡 已 profiling + numpy 列式过滤落地（过滤 83→4.4ms ~19×）；**Rust 暂不值得**，下一步 hnswlib（见 benchmarks/） |

---

## 17. 无历史包袱声明

旧 `python-nlql` 从未发布、无任何下游引用。**本项目即 v1，允许一切 breaking change**，一切以最佳设计为准，不做语法/接口兼容妥协。
- 沿用 `import nlql` 包名仅因简洁好用，非兼容考量。
- 旧实现（`_nlql_reference`）仅作缺陷参照，不作迁移目标。

---

_已落地的实现选型：核心依赖仅 `numpy + lark + httpx`（处处可装）；本地 ANN = 纯 numpy `FlatIndex`（精确、零原生依赖），`hnswlib/faiss` 为可选加速；embedder 三件套 = `FakeEmbedder`(确定性、供测试) / `OpenAIEmbedder`(HTTP、生产展示，已真机验证) / `SentenceTransformerEmbedder`(懒加载、可选本地)，均**不硬依赖、可重载**；Rust 加速预留不前置。NLQL/2 语法采用 LET 正交方案并已定稿。_
