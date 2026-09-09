"""Image OCR service shared by ingestion and the OCR API."""

from __future__ import annotations

from io import BytesIO
import shutil
from pathlib import Path
from typing import Any

from app.core.logging import get_logger
from app.core.config import settings
from app.utils.text import clean_text

log = get_logger(__name__)


class OCRUnavailable(RuntimeError):
    """Raised when the Python wrapper or native OCR engine is unavailable."""


def extract_text(image: bytes, language: str = "eng") -> dict[str, Any]:
    if not image:
        raise ValueError("Image cannot be empty")
    try:
        from PIL import Image, ImageEnhance, ImageFilter, ImageOps
        import pytesseract
        if settings.TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD
        elif not shutil.which("tesseract"):
            windows_tesseract = Path("C:/Program Files/Tesseract-OCR/tesseract.exe")
            if windows_tesseract.exists():
                pytesseract.pytesseract.tesseract_cmd = str(windows_tesseract)
    except ImportError as exc:
        raise OCRUnavailable(
            "OCR dependencies are unavailable. Install Pillow and pytesseract."
        ) from exc

    try:
        source = Image.open(BytesIO(image))
        source.load()
        grayscale = ImageOps.grayscale(source)
        enhanced = ImageEnhance.Contrast(grayscale).enhance(1.8)
        prepared = enhanced.filter(ImageFilter.SHARPEN)
        data = pytesseract.image_to_data(
            prepared,
            lang=language,
            config="--psm 6",
            output_type=pytesseract.Output.DICT,
        )
    except pytesseract.TesseractNotFoundError as exc:
        raise OCRUnavailable(
            "Tesseract OCR engine is not installed or is not on PATH. "
            "Set TESSERACT_CMD to its executable path."
        ) from exc
    except Exception as exc:
        raise ValueError("Could not decode or process the image") from exc

    words = []
    confidences = []
    for word, confidence in zip(data["text"], data["conf"]):
        word = word.strip()
        try:
            score = float(confidence)
        except (TypeError, ValueError):
            score = -1
        if word and score >= 0:
            words.append(word)
            confidences.append(score)
    text = clean_text(" ".join(words))
    return {
        "text": text,
        "confidence": round(sum(confidences) / len(confidences) / 100, 3)
        if confidences
        else 0.0,
        "word_count": len(words),
        "language": language,
    }
