# 查询语法

NLQL 是一条声明式查询，结构与 SQL 对应：`SELECT` 指定返回粒度，`LET` 计算相关度，`WHERE` 过滤，`ORDER BY` 与 `LIMIT` 排序限量。下面是一条覆盖主要子句的完整查询：

```sql
SELECT SPAN(SENTENCE, window => 1)
LET   relevance = SIMILARITY(content, "AI agent architecture"),
      novelty   = SIMILARITY(content, "novel unpublished ideas")
WHERE relevance >= 0.8
  AND meta.status != "draft"
  AND (content CONTAINS "planning" OR content CONTAINS "memory")
ORDER BY relevance DESC, novelty DESC
LIMIT 5
```

逐行解读：

- **`SELECT SPAN(SENTENCE, window => 1)`**——以句子为基本粒度，每个结果带左右各 1 个邻居句子作为上下文窗口
- **`LET relevance = SIMILARITY(...)`**——计算查询文本与单元内容的余弦相似度，绑定到名称 `relevance`
- **`LET novelty = ...`**——同一条查询可声明多个具名分数，互不干扰
- **`WHERE relevance >= 0.8`**——按分数过滤，引用上面绑定的名称
- **`meta.status != "draft"`**——按业务元数据过滤
- **`content CONTAINS "planning"`**——字符串包含判断，主语显式
- **`ORDER BY relevance DESC, novelty DESC`**——按分数排序，可多字段
- **`LIMIT 5`**——最多返回 5 条

## 子句一览

| 子句 | 作用 | 示例 |
|---|---|---|
| `SELECT` | 返回粒度 | `SELECT SENTENCE` / `SELECT CHUNK` / `SELECT SPAN(SENTENCE, window => 1)` |
| `LET` | 声明具名分数或计算值 | `LET rel = SIMILARITY(content, "x")` |
| `WHERE` | 过滤条件 | `WHERE rel >= 0.8 AND meta.status == "published"` |
| `ORDER BY` | 排序 | `ORDER BY rel DESC` |
| `LIMIT` | 限量 | `LIMIT 5` |

## 字段引用

查询中引用字段统一用点路径：

- **`content`**——单元的文本内容
- **`meta.字段名`**——文档的业务元数据，如 `meta.status`、`meta.created_at`
- **`vec.向量名`**——命名向量（多模态或多向量场景），如 `vec.image`
- **`relevance`** 等——`LET` 绑定的名称，可在 `WHERE` 与 `ORDER BY` 中复用

元数据走点路径，不再有专门的 `META("x")` 函数形态；`content` 与 `meta.status` 在语法上是同一种路径表达式。

## 函数

函数调用统一写成 `NAME(args)`，返回标量值。常见内置函数：

| 函数 | 返回 | 示例 |
|---|---|---|
| `SIMILARITY(content, "query")` | 数值 | 相关度分数，由召回阶段一次性计算 |
| `LENGTH(content)` | 数值 | `LENGTH(content)` 返回字符长度 |
| `COUNT(content, "word")` | 数值 | 子串出现次数 |
| `CONTAINS(content, "x")` | 布尔 | `content CONTAINS "x"`（中缀写法） |

`CONTAINS`、`MATCH`、`LIKE` 这类返回布尔的运算写成中缀只是语法糖，解析后与函数调用等价。新增函数通过 Registry 注册即可在查询中使用，无需改文法。

!!! note "关键字大小写"
    结构关键字（`SELECT` / `WHERE` / `LET` / `ORDER BY` / `AND` / `OR` / `NOT`）必须大写，且大小写敏感。这是为了与小写的元数据键（如 `meta.status`、`meta.window`）区分，避免冲突。

## 谓词与逻辑组合

谓词支持常见比较与中缀形式：

```sql
WHERE relevance >= 0.8
  AND meta.year > 2023
  AND content CONTAINS "agent"
  AND (meta.topic == "rag" OR meta.topic == "agents")
```

- 比较：`==`、`!=`、`>`、`>=`、`<`、`<=`
- 逻辑：`AND`、`OR`、`NOT`，括号分组
- 数值与日期按类型规约后比较（`meta.year > 2023` 是数值比较；ISO 日期按时间比较）

## 具名分数

`LET` 让一条查询携带多个语义分数，并在排序时复用：

```sql
LET   relevance = SIMILARITY(content, "AI agents"),
      novelty   = SIMILARITY(content, "novel ideas")
WHERE relevance >= 0.8
ORDER BY relevance DESC, novelty DESC
```

相同参数的 `SIMILARITY` 调用只计算一次，无论在 `LET` 中绑定还是内联在 `WHERE` 中。结果通过 `unit.scores["relevance"]` 读取。

## 相似度取值范围

`SIMILARITY` 返回原始余弦值，范围 `[-1, 1]`，不做 `(cos+1)/2` 折叠。因此阈值与直觉对应：高度相关接近 1，无关接近 0，语义相反为负。

## 下一步

- [三种写法](./three-entries.md)：把同样的查询改写成 Builder 或 JSON IR
- [数据模型](./data-model.md)：`content`、`meta.*` 背后的字段定义
- [快速上手](../tutorials/quickstart.md)：可运行的查询示例
- [API 参考 · lang](../../reference/lang.md)：解析器与文法细节
