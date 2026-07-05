"""Tests for document loaders and Engine.add_file."""

from __future__ import annotations

from pathlib import Path

import pytest

from nlql import Engine
from nlql.embed import FakeEmbedder
from nlql.embed import FakeMultimodalEmbedder
from nlql.errors import NLQLError
from nlql.loaders import PdfLoader, load_documents, register_loader
from nlql.model import Document, Modality


class TestTextLoader:
    def test_loads_text(self, tmp_path) -> None:
        path = tmp_path / "note.txt"
        path.write_text("AI agents plan. They use tools.", encoding="utf-8")
        docs = load_documents(path)
        assert len(docs) == 1
        assert docs[0].id == "note"
        assert "AI agents" in docs[0].payloads[0].as_text
        assert docs[0].metadata["source"].endswith("note.txt")

    def test_markdown_dispatch(self, tmp_path) -> None:
        path = tmp_path / "doc.md"
        path.write_text("# Title\n\nBody text.", encoding="utf-8")
        assert load_documents(path)[0].payloads[0].as_text.startswith("# Title")


class TestDocxLoader:
    def test_round_trip(self, tmp_path) -> None:
        docx = pytest.importorskip("docx")
        path = tmp_path / "report.docx"
        document = docx.Document()
        document.add_paragraph("First paragraph about AI agents.")
        document.add_paragraph("Second paragraph on memory and tools.")
        document.save(str(path))

        docs = load_documents(path)
        assert len(docs) == 1 and docs[0].id == "report"
        text = docs[0].payloads[0].as_text
        assert "AI agents" in text and "memory and tools" in text


class TestPdfLoader:
    def _make_pdf(self, path: Path, lines: list[str]) -> None:
        fpdf = pytest.importorskip("fpdf")
        pdf = fpdf.FPDF()
        for line in lines:
            pdf.add_page()
            pdf.set_font("helvetica", size=12)
            pdf.cell(0, 10, line)
        pdf.output(str(path))

    def test_extracts_text(self, tmp_path) -> None:
        pytest.importorskip("pypdf")
        path = tmp_path / "paper.pdf"
        self._make_pdf(path, ["Hello PDF world about vectors and agents."])
        docs = load_documents(path)
        assert len(docs) == 1
        assert "PDF world" in docs[0].payloads[0].as_text

    def test_by_page(self, tmp_path) -> None:
        pytest.importorskip("pypdf")
        path = tmp_path / "multi.pdf"
        self._make_pdf(path, ["Page one content here.", "Page two content here."])
        docs = list(PdfLoader(by_page=True).load(path))
        assert len(docs) == 2
        assert docs[0].metadata["page"] == 1
        assert docs[0].id == "multi#p1"


class TestDispatchAndEngine:
    def test_unknown_extension_raises(self, tmp_path) -> None:
        path = tmp_path / "file.xyz"
        path.write_text("x", encoding="utf-8")
        with pytest.raises(NLQLError):
            load_documents(path)

    def test_register_custom_loader(self, tmp_path) -> None:
        class UpperLoader:
            def load(self, source, *, metadata=None):
                yield Document.from_text(
                    Path(source).read_text(encoding="utf-8").upper(), id="x", metadata=metadata or {}
                )

        register_loader(UpperLoader(), ".up")
        path = tmp_path / "a.up"
        path.write_text("hello", encoding="utf-8")
        assert load_documents(path)[0].payloads[0].as_text == "HELLO"

    def test_engine_add_file(self, tmp_path) -> None:
        path = tmp_path / "kb.txt"
        path.write_text("Agents plan and act. They call external tools.", encoding="utf-8")
        engine = Engine(FakeEmbedder())
        ids = engine.add_file(str(path))
        assert ids == ["kb"]
        results = engine.search('SELECT SENTENCE WHERE content CONTAINS "tools"')
        assert len(results) >= 1


class TestPdfImageExtraction:
    def _pdf_with_image(self, tmp_path: Path) -> Path:
        pytest.importorskip("pypdf")
        fpdf = pytest.importorskip("fpdf")
        Image = pytest.importorskip("PIL.Image")
        img_path = tmp_path / "figure.png"
        Image.new("RGB", (48, 48), (200, 60, 60)).save(str(img_path))
        pdf = fpdf.FPDF()
        pdf.add_page()
        pdf.set_font("helvetica", size=12)
        pdf.cell(0, 10, "A report page with an embedded figure.")
        pdf.image(str(img_path), x=10, y=30, w=20)
        pdf_path = tmp_path / "report.pdf"
        pdf.output(str(pdf_path))
        return pdf_path

    def test_images_become_payloads(self, tmp_path) -> None:
        pdf_path = self._pdf_with_image(tmp_path)
        docs = list(PdfLoader(extract_images=True).load(pdf_path))
        assert len(docs) == 1
        modalities = {p.modality for p in docs[0].payloads}
        assert Modality.IMAGE in modalities
        assert Modality.TEXT in modalities
        assert docs[0].metadata["images"] >= 1

    def test_images_ingest_as_units(self, tmp_path) -> None:
        pdf_path = self._pdf_with_image(tmp_path)
        engine = Engine(FakeMultimodalEmbedder(), granularity="chunk")
        engine.add_file(str(pdf_path), loader=PdfLoader(extract_images=True))
        image_units = [u for u in engine.store.all_units() if u.payload.modality is Modality.IMAGE]
        assert len(image_units) >= 1  # the figure became an image unit alongside the text
