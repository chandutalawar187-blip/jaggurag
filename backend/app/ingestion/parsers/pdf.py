"""PDF parser — extracts text page-by-page using PyMuPDF."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from app.core.logging import get_logger
from app.services.ocr_service import OCRUnavailable, extract_text
from app.utils.text import clean_text

log = get_logger(__name__)


def parse(filepath: str | Path) -> List[Tuple[str, Dict[str, Any]]]:
    """Return list of (text, metadata) tuples, one per page."""
    try:
        import fitz  # PyMuPDF <= 1.27
    except ModuleNotFoundError:
        import pymupdf as fitz  # PyMuPDF >= 1.28

    filepath = Path(filepath)
    results: List[Tuple[str, Dict[str, Any]]] = []

    try:
        doc = fitz.open(str(filepath))
        for page_num in range(len(doc)):
            page = doc[page_num]

            # Extract text
            text = page.get_text("text")

            # Also try to extract tables as structured text
            tables = page.find_tables()
            table_text = ""
            if tables and tables.tables:
                for table in tables.tables:
                    try:
                        for row in table.extract():
                            cells = [str(c).strip() if c else "" for c in row]
                            if any(cells):
                                table_text += " | ".join(cells) + "\n"
                        table_text += "\n"
                    except Exception:
                        pass

            combined = clean_text(text)
            if table_text.strip():
                combined += "\n\n[TABLE]\n" + clean_text(table_text)

            # Scanned PDFs often contain only a watermark in the text layer.
            # OCR the rendered page when native extraction has no useful text.
            if len(combined.strip()) < 40:
                try:
                    pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                    ocr_result = extract_text(pixmap.tobytes("png"))
                    ocr_text = clean_text(ocr_result["text"])
                    if len(ocr_text) > len(combined.strip()):
                        combined = ocr_text
                        source_type = "pdf_ocr"
                    else:
                        source_type = "pdf"
                except OCRUnavailable:
                    source_type = "pdf"
            else:
                source_type = "pdf"

            if combined.strip():
                results.append((combined, {
                    "page_number": page_num + 1,
                    "source_type": source_type,
                }))
        doc.close()
    except Exception as e:
        log.error("PDF parse error for %s: %s", filepath.name, e)
        raise

    if not results:
        # Try plain text extraction as fallback
        try:
            doc = fitz.open(str(filepath))
            full_text = ""
            for page in doc:
                full_text += page.get_text() + "\n"
            doc.close()
            if full_text.strip():
                results.append((clean_text(full_text), {"page_number": 1, "source_type": "pdf"}))
        except Exception:
            pass

    log.info("PDF parsed: %s → %d pages", filepath.name, len(results))
    return results
