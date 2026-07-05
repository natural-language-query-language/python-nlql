"""Generate /llms.txt and /llms-full.txt from the default-language (zh) docs.

Reads ``docs/zh/`` source files directly. This bypasses mkdocs-static-i18n,
which ``mkdocs-llmstxt`` is incompatible with in folder mode (it cannot match
the per-language page URIs).
"""

from __future__ import annotations

from pathlib import Path

DESC = (
    "NLQL 是一个语义检索工具，用类似 SQL 的语句从文本中查找相关内容，"
    "适合 Agent 与 RAG 应用。NLQL 语句、Python 链式构造、LLM 工具调用"
    "三种写法编译到同一 IR，结果一致。支持 Qdrant、Faiss、Chroma、"
    "HnswLib、pgvector 等可插拔后端，文本与图像的多模态检索，"
    "以及 engine.explain() 执行计划输出。"
)

# docs/hooks.py -> docs/zh/  (default-language source)
ZH = Path(__file__).resolve().parent / "zh"


def _title(p: Path) -> str:
    """First H1 in the file, fallback to a humanized filename."""
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
    except OSError:
        pass
    return p.stem.replace("-", " ").replace("_", " ")


def on_post_build(config, **kwargs):  # noqa: ANN001, ANN401
    site = Path(config["site_dir"])
    pages = sorted(ZH.rglob("*.md"))
    base = (config.get("site_url") or "/").rstrip("/")

    # llms.txt — concise index of pages
    lines = ["# NLQL", "", f"> {DESC}", "", DESC, ""]
    for p in pages:
        rel = p.relative_to(ZH).with_suffix("")
        url = f"{base}/{'/'.join(rel.parts)}/"
        lines.append(f"- [{_title(p)}]({url})")
    (site / "llms.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # llms-full.txt — concatenated source markdown
    full = ["# NLQL", "", DESC, ""]
    for p in pages:
        full.append(p.read_text(encoding="utf-8").rstrip())
    (site / "llms-full.txt").write_text("\n\n".join(full) + "\n", encoding="utf-8")
