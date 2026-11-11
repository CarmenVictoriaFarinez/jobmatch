"""Lectura de CV a texto plano. Sin LLM y sin OCR."""

from __future__ import annotations

import io
from pathlib import Path

from docx import Document
from pypdf import PdfReader


class ReadError(ValueError):
    """Archivo ilegible o formato no soportado."""


def read_cv_file(data: bytes, filename: str) -> tuple[str, list[str]]:
    """Devuelve (texto, avisos). El texto puede estar vacío; entonces hay aviso."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(data)
    if suffix == ".docx":
        return _read_docx(data)
    raise ReadError(
        f"Formato no soportado: {suffix or filename}. Usa PDF con texto seleccionable o DOCX."
    )


def _read_pdf(data: bytes) -> tuple[str, list[str]]:
    warnings: list[str] = []
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001 — mostrar error usable en la UI
        raise ReadError(f"No se pudo abrir el PDF: {exc}") from exc

    if getattr(reader, "is_encrypted", False):
        raise ReadError("El PDF está protegido con contraseña.")

    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    text = "\n".join(pages).strip()

    if not text:
        warnings.append(
            "El PDF no tiene texto seleccionable. Fase 1 no hace OCR; "
            "exporta el CV a PDF de texto o súbelo en DOCX."
        )
    return text, warnings


def _read_docx(data: bytes) -> tuple[str, list[str]]:
    try:
        document = Document(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001
        raise ReadError(f"No se pudo abrir el DOCX: {exc}") from exc

    parts = [p.text for p in document.paragraphs if p.text and p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    text = "\n".join(parts).strip()
    warnings: list[str] = []
    if not text:
        warnings.append("El DOCX no contiene párrafos ni celdas con texto.")
    return text, warnings
