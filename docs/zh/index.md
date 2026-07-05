# NLQL

NLQL 是一个语义检索工具：用类似 SQL 的语句，从文本中查找相关内容。

查询的规范形式是一份可序列化的中间表示（IR）。NLQL 字符串、Python 链式构造、LLM 工具调用三种写法都编译到同一份 IR，因此结果一致。

## 示例

```sql
SELECT SENTENCE
LET   relevance = SIMILARITY(content, "AI Agent 架构"),
      novelty   = SIMILARITY(content, "新颖的未发表想法")
WHERE relevance >= 0.8
  AND meta.status != "draft"
ORDER BY relevance DESC, novelty DESC
LIMIT 5
```

语句结构与 SQL 对应：`SELECT` 指定返回粒度，`LET` 计算相关度，`WHERE` 过滤，`ORDER BY` 与 `LIMIT` 排序限量。相关度计算、过滤、排序集中在一条语句中，无需分散到业务代码。

## 三种写法

同一查询的三种写法，结果一致：

=== "NLQL 语句"

    ```python
    engine.search('''
        SELECT SENTENCE
        LET rel = SIMILARITY(content, "autonomous agents")
        WHERE rel >= 0.8 AND meta.status == "published"
        ORDER BY rel DESC
        LIMIT 5
    ''')
    ```

=== "Python 链式"

    ```python
    from nlql.sdk.builder import select, similarity, F, Meta

    query = (select("sentence")
        .let("rel", similarity("content", "autonomous agents"))
        .where((F("rel") >= 0.8) & (Meta("status") == "published"))
        .order_by("rel", desc=True)
        .limit(5)
        .build())

    engine.search(query)
    ```

=== "LLM 工具调用"

    ```python
    schema = engine.function_tool()           # 工具描述，交给 LLM
    engine.search_ir({ "select": ..., ... })  # 执行 LLM 返回的查询
    ```

## 安装

```bash
pip install python-nlql              # 核心
pip install "python-nlql[faiss]"     # 可选：Faiss 后端（另有 hnsw / qdrant / chroma / pgvector）
pip install "python-nlql[loaders]"   # 可选：加载 DOCX / PDF
```

## 特性

- **声明式查询**：相关度、过滤、排序集中在一条语句
- **后端可插拔**：内置存储开箱即用；切换 Qdrant、Faiss 等仅改动一行
- **多模态**：文本与图像在同一向量空间，可用文字检索图像
- **可解释**：`engine.explain()` 输出查询的执行计划

## 下一步

- [快速开始](content/tutorials/quickstart.md)
- [设计思路](content/concepts/overview.md)
- [混合后端](content/tutorials/hybrid-stores.md)
- [API 参考](reference/sdk.md)
