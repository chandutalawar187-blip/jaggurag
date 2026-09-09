"""Image parser — extracts text via OCR using pytesseract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from app.core.logging import get_logger
from app.services.ocr_service import OCRUnavailable, extract_text

log = get_logger(__name__)


def parse(filepath: str | Path) -> List[Tuple[str, Dict[str, Any]]]:
    filepath = Path(filepath)

    try:
        result = extract_text(filepath.read_bytes())
        cleaned = result["text"]

        if not cleaned or len(cleaned) < 10:
            return [(cleaned or "[Image with minimal text]", {
                "source_type": "image",
                "ocr_confidence": result["confidence"],
            })]

        return [(cleaned, {
            "source_type": "image",
            "ocr_confidence": result["confidence"],
        })]
    except OCRUnavailable:
        log.warning("OCR unavailable for %s", filepath.name)
        raise
    except Exception as e:
        log.error("Image parse error for %s: %s", filepath.name, e)
        raise
