"""Provider interfaces for handwriting OCR and embeddings."""

from __future__ import annotations

from dataclasses import dataclass
import base64
from typing import Any, Protocol

import httpx

from app.core.config import settings
from app.services.embedding_service import EmbeddingService
from app.services.ocr_service import extract_text


@dataclass
class OCRPage:
    text: str
    raw_text: str
    confidence: float | None
    bounding_boxes: list[dict[str, Any]]


class OCRProvider(Protocol):
    def extract_text(self, image: bytes) -> OCRPage:
        ...


class TesseractOCRProvider:
    """Default local OCR provider; confidence comes from Tesseract output."""

    def extract_text(self, image: bytes) -> OCRPage:
        result = extract_text(image)
        return OCRPage(
            text=result["text"],
            raw_text=result["text"],
            confidence=result.get("confidence"),
            bounding_boxes=[],
        )


class QwenVisionOCRProvider:
    """Qwen-compatible multimodal OCR provider with optional metadata."""

    def extract_text(self, image: bytes) -> OCRPage:
        if not settings.OCR_RAG_VISION_MODEL:
            raise RuntimeError("OCR_RAG_VISION_MODEL is not configured")
        encoded = base64.b64encode(image).decode("ascii")
        response = httpx.post(
            f"{settings.OLLAMA_BASE_URL}/chat/completions",
            json={
                "model": settings.OCR_RAG_VISION_MODEL,
                "temperature": 0,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Transcribe this handwritten page exactly. Return only the text."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                    ],
                }],
            },
            timeout=120,
        )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"].strip()
        return OCRPage(text=text, raw_text=text, confidence=None, bounding_boxes=[])


def get_ocr_provider() -> OCRProvider:
    if settings.OCR_RAG_PROVIDER.lower() in {"qwen", "qwen_vision", "vision"}:
        return QwenVisionOCRProvider()
    return TesseractOCRProvider()


class EmbeddingProvider:
    def __init__(self) -> None:
        self._service = EmbeddingService()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._service.embed_batch(texts)
