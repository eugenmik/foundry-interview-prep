"""Extract plain text from an uploaded resume (PDF / DOCX / TXT)."""

from __future__ import annotations

import io


def extract_text(filename: str, data: bytes) -> str:
    """Return plain text extracted from an uploaded file's bytes.

    Supports PDF, DOCX and plain text. Raises ValueError for unsupported
    types and RuntimeError if parsing fails.
    """
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        return _from_pdf(data)
    if name.endswith(".docx"):
        return _from_docx(data)
    if name.endswith(".txt") or name.endswith(".md"):
        return data.decode("utf-8", errors="ignore")
    raise ValueError("Unsupported file type. Use PDF, DOCX or TXT.")


def _from_pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pypdf is required to read PDF files.") from exc
    reader = PdfReader(io.BytesIO(data))
    parts = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(parts).strip()


def _from_docx(data: bytes) -> str:
    try:
        import docx  # python-docx
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("python-docx is required to read DOCX files.") from exc
    document = docx.Document(io.BytesIO(data))
    return "\n".join(p.text for p in document.paragraphs).strip()
