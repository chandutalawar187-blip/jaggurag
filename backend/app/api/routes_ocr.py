"""OCR extraction endpoint."""

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.services.ocr_service import OCRUnavailable, extract_text
from app.services.document_service import upload_document

router = APIRouter()


@router.post("/ocr/extract")
async def extract_ocr(
    image: UploadFile = File(...),
    language: str = Form("eng"),
):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")
    try:
        content = await image.read()
        result = extract_text(content, language=language)
        indexed = await upload_document(image.filename or "ocr-document.png", content)
        return {"filename": indexed["filename"], **result, **indexed}
    except OCRUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
