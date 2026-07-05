"""Document loaders. Importing this package registers the built-in txt/md/docx/pdf loaders."""

from nlql.loaders.base import Loader, load_documents, loader_for, register_loader
from nlql.loaders.docx_loader import DocxLoader
from nlql.loaders.pdf_loader import PdfLoader
from nlql.loaders.text import TextLoader

register_loader(TextLoader(), ".txt", ".md", ".markdown", ".text")
register_loader(DocxLoader(), ".docx")
register_loader(PdfLoader(), ".pdf")

__all__ = [
    "Loader",
    "load_documents",
    "register_loader",
    "loader_for",
    "TextLoader",
    "DocxLoader",
    "PdfLoader",
]
