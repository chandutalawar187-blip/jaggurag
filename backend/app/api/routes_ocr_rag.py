"""API routes for the isolated handwritten OCR-RAG feature."""

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.ocr_rag.service import process_document, query_document, upload_handwritten
from app.ocr_rag.storage import get_document

router = APIRouter(prefix="/ocr-rag")


class OCRRAGQuery(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000)
    top_k: int = Field(default=5, ge=1, le=10)


@router.post("/documents")
async def upload_ocr_rag_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    try:
        document = await upload_handwritten(file.filename, await file.read())
        background_tasks.add_task(
            process_document,
            document["id"],
            __import__("pathlib").Path(document["original_path"]),
        )
        return document
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@router.get("/documents/{document_id}")
async def get_ocr_rag_document(document_id: str):
    document = get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="OCR-RAG document not found")
    return document


@router.post("/documents/{document_id}/query")
async def query_ocr_rag_document(document_id: str, request: OCRRAGQuery):
    try:
        return query_document(document_id, request.question, request.top_k)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail="OCR-RAG query failed") from error
