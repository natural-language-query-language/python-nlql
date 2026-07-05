# NLQL

[![PyPI version](https://img.shields.io/pypi/v/python-nlql.svg?label=pypi)](https://pypi.org/project/python-nlql/)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-online-blue.svg)](https://natural-language-query-language.github.io/python-nlql/)

[English](README.md) · **简体中文** · [在线文档](https://natural-language-query-language.github.io/python-nlql/)

NLQL 让你用类似 SQL 的语句做语义检索。把"从文本里找相关内容"这件事，变得像查数据库一样直接——相关度计算、过滤、排序写在一条查询里，不再需要拼凑 embedding 调用和后处理代码。

适合 Agent 与 RAG 应用：查询本身就是结构化数据，可以直接作为大模型的工具调用载体。

## 它长什么样

```python
import nlql

engine = nlql.Engine(nlql.embed.FakeEmbedder())  # 或 OpenAIEmbedder，以及任意 Embedder 实现
engine.add_text("AI agents plan tasks and call tools.", metadata={"status": "published"})
engine.add_text("Banana bread needs flour and sugar.", metadata={"status": "draft"})

for unit in engine.search('''
    SELECT SENTENCE
    LET rel = SIMILARITY(content, "autonomous agents")
    WHERE rel >= 0.2 AND meta.status == "published"
    ORDER BY rel DESC
    LIMIT 5
'''):
    print(f"{unit.scores['rel']:+.3f}  {unit.content}")
```

语句和 SQL 几乎一样：`SELECT` 指定返回粒度，`LET` 算相关度，`WHERE` 过滤，`ORDER BY` / `LIMIT` 排序限量。

## 特性

- **一条语句表达完整意图** —— 相关度、过滤、排序集中在一处，不再散在业务代码里
- **三种写法，结果一致** —— SQL 语句、Python 链式构造、JSON IR，都编译到同一份内部表示
- **后端可插拔** —— 内置存储开箱即用；切换 Qdrant / Faiss / Chroma / HnswLib / pgvector 只需改一行
- **召回 + 重排两段式** —— 向量召回后挂重排器，提升结果准确性
- **多模态** —— 文本与图像在同一向量空间，用文字检索图像
- **可解释** —— `engine.explain()` 输出查询的执行计划

## 安装

```bash
pip install python-nlql
```

可选依赖：

| 命令 | 用途 |
|---|---|
| `pip install "python-nlql[faiss]"` | Faiss 后端 |
| `pip install "python-nlql[hnsw]"` | HnswLib 后端（适合大数据量） |
| `pip install "python-nlql[qdrant]"` | Qdrant 后端 |
| `pip install "python-nlql[chroma]"` | Chroma 后端 |
| `pip install "python-nlql[pgvector]"` | Postgres + pgvector 后端 |
| `pip install "python-nlql[local]"` | 本地 sentence-transformers / CLIP / cross-encoder |
| `pip install "python-nlql[loaders]"` | 加载 DOCX / PDF 文件 |

## 切换后端

切换存储后端只需一行，写入与查询代码完全不变：

```python
from nlql.store.qdrant_store import QdrantStore
engine = nlql.Engine(embedder, store=QdrantStore(location=":memory:"))
```

## 文档

完整文档、教程与 API 参考：**https://natural-language-query-language.github.io/python-nlql/**

- [快速开始](https://natural-language-query-language.github.io/python-nlql/content/tutorials/quickstart/)
- [设计思路](https://natural-language-query-language.github.io/python-nlql/content/concepts/overview/)
- [API 参考](https://natural-language-query-language.github.io/python-nlql/reference/sdk/)
- [English docs](https://natural-language-query-language.github.io/python-nlql/en/)

更多示例见 [`examples/`](examples/) 目录。

## License

[MIT](LICENSE)
