"""Load documents from files (txt / md / docx / pdf) and query them.

Run: python examples/document_loading.py

Creates a few files in a temp dir, ingests them with ``add_files`` (dispatch by extension),
and searches. DOCX needs ``pip install python-nlql[loaders]``; here it is added only if available.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import nlql
from nlql.embed import FakeEmbedder


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    (tmp / "agents.txt").write_text(
        "AI agents plan tasks. They keep memory and call external tools.", encoding="utf-8"
    )
    (tmp / "rag.md").write_text(
        "# RAG\n\nRetrieval-augmented generation grounds LLM answers in your documents.",
        encoding="utf-8",
    )
    files = [tmp / "agents.txt", tmp / "rag.md"]

    try:  # add a .docx too if python-docx is installed
        import docx

        document = docx.Document()
        document.add_paragraph("Vector databases store embeddings for similarity search.")
        document.save(str(tmp / "vectors.docx"))
        files.append(tmp / "vectors.docx")
    except ImportError:
        print("(python-docx not installed — skipping the .docx file)")

    engine = nlql.Engine(FakeEmbedder())
    ids = engine.add_files([str(f) for f in files])
    print(f"loaded {len(ids)} files -> {len(engine)} sentence units: {ids}\n")

    results = engine.search(
        'SELECT SENTENCE LET rel = SIMILARITY(content, "how do agents use tools") '
        "ORDER BY rel DESC LIMIT 3"
    )
    for unit in results:
        print(f"  ({unit.doc_id}) {unit.content}")


if __name__ == "__main__":
    main()
