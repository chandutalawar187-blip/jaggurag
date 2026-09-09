"""Handwritten document OCR -> chunk -> embedding -> retrieval -> Qwen service."""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.core.security import sanitize_filename, validate_file_extension, validate_file_size
from app.models.database import get_chroma_client
from app.services.document_service import upload_document
from app.services.lmstudio_service import generate
from app.ocr_rag.providers import EmbeddingProvider, OCRPage, get_ocr_provider
from app.ocr_rag.storage import (
    get_document,
    init_storage,
    save_pages,
    update_document,
    upsert_document,
)

log = get_logger(__name__)


def _collection():
    return get_chroma_client().get_or_create_collection(
        name=settings.OCR_RAG_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def _clean_text(value: str) -> str:
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", value)).strip()


def _render_pages(path: Path) -> list[tuple[int, bytes, str]]:
    if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        return [(1, path.read_bytes(), str(path))]
    try:
        import fitz
    except ModuleNotFoundError:
        import pymupdf as fitz
    document = fitz.open(str(path))
    pages = []
    try:
        for index, page in enumerate(document):
            pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
            pages.append((index + 1, pixmap.tobytes("png"), f"{path}#page={index + 1}"))
    finally:
        document.close()
    return pages


def _chunks(document_id: str, page: dict[str, Any]) -> list[dict[str, Any]]:
    paragraphs = [part.strip() for part in page["text"].split("\n\n") if part.strip()]
    if not paragraphs and page["text"].strip():
        paragraphs = [page["text"].strip()]
    result = []
    for index, text in enumerate(paragraphs):
        result.append({
            "id": f"{document_id}:{page['page_number']}:{index}:{uuid.uuid4().hex[:8]}",
            "text": text,
            "metadata": {
                "ocr_rag": True,
                "document_id": document_id,
                "page_number": page["page_number"],
                "ocr_confidence": page.get("confidence") if page.get("confidence") is not None else -1.0,
                "image_reference": page["image_reference"],
                "bounding_boxes": str(page.get("bounding_boxes", [])),
            },
        })
    return result


def process_document(document_id: str, path: Path) -> None:
    provider = get_ocr_provider()
    pages: list[dict[str, Any]] = []
    try:
        rendered_pages = _render_pages(path)
        total_pages = len(rendered_pages)
        update_document(document_id, progress=5, progress_phase="rendering")
        for page_index, (page_number, image, image_reference) in enumerate(rendered_pages, 1):
            result: OCRPage = provider.extract_text(image)
            text = _clean_text(result.text)
            if not text:
                continue
            pages.append({
                "id": f"{document_id}:page:{page_number}",
                "page_number": page_number,
                "text": text,
                "raw_text": result.raw_text,
                "confidence": result.confidence,
                "image_reference": image_reference,
                "bounding_boxes": result.bounding_boxes,
            })
            update_document(
                document_id,
                progress=5 + round(page_index / max(total_pages, 1) * 55),
                progress_phase="ocr",
                page_count=page_index,
            )

        if not pages:
            raise ValueError("No readable handwritten text was extracted")

        chunks = [chunk for page in pages for chunk in _chunks(document_id, page)]
        update_document(document_id, progress=65, progress_phase="chunking")
        embeddings = EmbeddingProvider().embed([chunk["text"] for chunk in chunks])
        update_document(document_id, progress=85, progress_phase="embedding")
        collection = _collection()
        collection.delete(where={"ocr_rag_document_id": document_id})
        collection.add(
            ids=[chunk["id"] for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk["text"] for chunk in chunks],
            metadatas=[
                {**chunk["metadata"], "ocr_rag_document_id": document_id}
                for chunk in chunks
            ],
        )
        save_pages(document_id, pages)
        update_document(
            document_id, status="completed", page_count=len(pages),
            chunk_count=len(chunks), progress=100, progress_phase="completed",
            error_message="",
        )
        log.info("OCR-RAG indexed document=%s pages=%d chunks=%d", document_id, len(pages), len(chunks))
    except Exception as exc:
        log.error("OCR-RAG processing failed document=%s: %s", document_id, exc)
        update_document(
            document_id, status="failed", progress_phase="failed",
            error_message=str(exc)[:500],
        )


async def upload_handwritten(filename: str, content: bytes) -> dict[str, Any]:
    init_storage()
    safe_name = sanitize_filename(filename)
    if not validate_file_extension(safe_name):
        raise ValueError(f"Unsupported file type: {Path(safe_name).suffix}")
    if not validate_file_size(len(content)):
        raise ValueError(f"File too large. Maximum: {settings.MAX_UPLOAD_SIZE_MB} MB")
    if Path(safe_name).suffix.lower() not in {".pdf", ".png", ".jpg", ".jpeg"}:
        raise ValueError("Handwritten OCR-RAG supports PDF, PNG, JPG, and JPEG files")

    # Reuse the existing secure upload and persistence path.
    indexed = await upload_document(safe_name, content)
    document_id = uuid.uuid4().hex
    original_path = Path(settings.DATASET_PATH) / indexed["filename"]
    upsert_document({
        "id": document_id, "filename": indexed["filename"],
        "original_path": str(original_path), "status": "processing",
    })
    document = get_document(document_id)
    if not document:
        raise RuntimeError("OCR-RAG document status could not be persisted")
    return document


def query_document(document_id: str, question: str, top_k: int = 5) -> dict[str, Any]:
    document = get_document(document_id)
    if not document:
        raise LookupError("OCR-RAG document not found")
    if document["status"] != "completed":
        raise RuntimeError(f"OCR-RAG document is not ready: {document['status']}")
    query_embedding = EmbeddingProvider().embed([question])[0]
    result = _collection().query(
        query_embeddings=[query_embedding],
        n_results=max(1, min(top_k, 10)),
        where={"ocr_rag_document_id": document_id},
        include=["documents", "metadatas", "distances"],
    )
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    result_ids = result.get("ids", [[]])[0]
    if _is_induction_definition(question):
        opening = _collection().get(
            where={"ocr_rag_document_id": document_id},
            include=["documents", "metadatas"],
        )
        opening_rows = [
            (text, meta, chunk_id)
            for text, meta, chunk_id in zip(
                opening.get("documents", []), opening.get("metadatas", []),
                opening.get("ids", []),
            )
            if meta.get("page_number") in (1, "1", 2, "2")
        ]
        if opening_rows:
            documents = [row[0] for row in opening_rows[:top_k]]
            metadatas = [row[1] for row in opening_rows[:top_k]]
            result_ids = [row[2] for row in opening_rows[:top_k]]
    evidence = "\n\n".join(
        f"[Page {meta.get('page_number')}] {text}"
        for text, meta in zip(documents, metadatas)
    )
    if not evidence:
        return {"answer": "The handwritten document does not contain enough readable information to answer this question.", "sources": []}
    prompt = (
        "Answer only from the handwritten document evidence below. Do not invent "
        "names, numbers, dates, or amounts. If unsupported, say so clearly.\n\n"
        f"DOCUMENT EVIDENCE:\n{evidence}\n\nQUESTION:\n{question}\n"
        "Return a concise answer and do not cite pages not present in the evidence."
    )
    answer = generate(prompt, model_name="local_qwen")
    sources = [
        {"page": meta.get("page_number"), "chunk_id": chunk_id}
        for meta, chunk_id in zip(metadatas, result_ids)
        if meta.get("page_number") is not None
    ]
    return {"answer": answer, "sources": sources}


def _is_induction_definition(question: str) -> bool:
    normalized = question.lower()
    return "induction" in normalized and any(
        phrase in normalized
        for phrase in ("what is", "define", "definition", "explain", "principle")
    )
