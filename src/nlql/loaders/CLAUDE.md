[根目录 CLAUDE.md](../../CLAUDE.md) > [src](../) > [nlql](.) > **loaders**

# loaders — 文档加载器

## 模块职责

把外部文件（`.txt` / `.md` / `.docx` / `.pdf`）加载为 `Document`，再走统一写入管线。按扩展名分派；可 `register_loader` 扩展更多格式。

## 入口与对外接口

```python
from nlql.loaders import Loader, load_documents, register_loader, loader_for, TextLoader, DocxLoader, PdfLoader

engine.add_file("report.docx")                # 自动分派
engine.add_files(["a.pdf", "b.md", "c.txt"])  # 批量
register_loader(MyEpubLoader(), ".epub")      # 扩展
engine.add_file(path, loader=PdfLoader(by_page=True))  # 显式
```

- `Loader` Protocol：`.load(path) -> list[Document]`（一份文件可产多 Document，如 PDF 逐页或抽图）。
- `load_documents(path, loader=None, metadata=None)` — 入口；`loader=None` 时按扩展名查 `loader_for(ext)`。
- `register_loader(loader, *exts)` — 注册；导入包即副作用注册内置 txt/md/docx/pdf。
- `PdfLoader(extract_images=True)` 可从 PDF 抽图作为 image payload。

## 关键依赖与配置

- 可选 extras（懒导入）：`python-docx>=1.1`（`nlql[loaders]`，DOCX）/ `pypdf>=4.0`（PDF）。
- 上游：`model.Document`；下游：`sdk.Engine.add_file/add_files`。

## 测试与质量

- `tests/test_loaders.py`（分派 / 注册 / 各格式加载）。

## 相关文件清单

- [`base.py`](base.py) — `Loader` Protocol / `load_documents` / `register_loader` / `loader_for`
- [`text.py`](text.py) — `TextLoader`（.txt/.md/.markdown/.text）
- [`docx_loader.py`](docx_loader.py) — `DocxLoader`
- [`pdf_loader.py`](pdf_loader.py) — `PdfLoader`

## 变更记录

- 2026-07-05：首次生成。
