# 文档加载

`engine.add_file` 与 `engine.add_files` 按文件扩展名自动选择加载器，把文件转成 `Document` 后写入索引。内置支持 `.txt` / `.md` / `.docx` / `.pdf`，可通过 `register_loader` 扩展更多格式。

本例用 `FakeEmbedder` 离线演示。

```python
import tempfile
from pathlib import Path
import nlql

tmp = Path(tempfile.mkdtemp())
(tmp / "agents.txt").write_text(
    "AI agents plan tasks. They keep memory and call external tools.", encoding="utf-8"
)
(tmp / "rag.md").write_text(
    "# RAG\n\nRetrieval-augmented generation grounds LLM answers in your documents.",
    encoding="utf-8",
)
files = [tmp / "agents.txt", tmp / "rag.md"]
```

## 按扩展名加载

`add_files` 接收路径列表，内部对每个文件按扩展名分派加载器。`.txt` 与 `.md` 走纯文本加载器，开箱即用。

```python
engine = nlql.Engine(nlql.FakeEmbedder())
ids = engine.add_files([str(f) for f in files])
print(f"loaded {len(ids)} files -> {len(engine)} sentence units: {ids}")
```

返回值是写入的文档 id 列表。每个文件可能产出多个 `Document`（例如 PDF 逐页），因此返回的是列表。

## 加载 DOCX / PDF

DOCX 与 PDF 依赖额外的解析库，安装后即可加载：

```bash
pip install "python-nlql[loaders]"
```

下面这段代码在 `python-docx` 可用时加入一份 `.docx`，否则跳过：

```python
try:
    import docx

    document = docx.Document()
    document.add_paragraph("Vector databases store embeddings for similarity search.")
    document.save(str(tmp / "vectors.docx"))
    files.append(tmp / "vectors.docx")
except ImportError:
    print("(python-docx not installed — skipping the .docx file)")

engine = nlql.Engine(nlql.FakeEmbedder())
ids = engine.add_files([str(f) for f in files])
```

`add_file` 加载单个文件，返回该文件的文档 id 列表：

```python
ids = engine.add_file("report.pdf", metadata={"source": "annual-report"})
```

PDF 加载器支持 `PdfLoader(extract_images=True)` 从文件中抽取图像载荷。

## 显式指定加载器

`add_file(path, loader=...)` 可绕过自动分派，直接用某个加载器：

```python
from nlql.loaders import PdfLoader

engine.add_file("scan.pdf", loader=PdfLoader(by_page=True))
```

## 注册自定义格式

`register_loader` 把一个实现 `Loader` 协议（`load(path) -> list[Document]`）的加载器绑定到一个或多个扩展名。导入 `nlql.loaders` 即注册了内置格式。

```python
from nlql.loaders import register_loader, TextLoader

# 让 .log 也按纯文本加载
register_loader(TextLoader(), ".log")

# 自定义加载器
class EpubLoader:
    def load(self, path):
        ...
        return [nlql.Document.from_text(text, ...)]

register_loader(EpubLoader(), ".epub")
```

注册后 `engine.add_file("book.epub")` 即可走自定义逻辑。

!!! note "DOCX / PDF 需要安装 extras"
    纯文本格式无需额外依赖；加载 `.docx` 与 `.pdf` 前请安装 `python-nlql[loaders]`，否则会抛 `ImportError`。

## 查询

写入完成后查询与其它入口一致：

```python
results = engine.search(
    'SELECT SENTENCE LET rel = SIMILARITY(content, "how do agents use tools") '
    "ORDER BY rel DESC LIMIT 3"
)
for unit in results:
    print(f"  ({unit.doc_id}) {unit.content}")
```

## 下一步

- [快速开始](./quickstart.md)
- [多模态检索](./multimodal-search.md)
- [Document 与 Unit 数据模型](../concepts/overview.md)
- [Loader API](../../reference/loaders.md)

---

**完整源码**：[`examples/document_loading.py`](https://github.com/natural-language-query-language/python-nlql/blob/main/examples/document_loading.py)
