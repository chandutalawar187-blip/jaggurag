from pathlib import Path
try:
    import fitz
except ImportError:
    try:
        import pymupdf as fitz
    except ImportError:
        fitz = None

class PDFLoader:
    def __init__(self, dpi=144, image_format="png"):
        self.dpi, self.image_format = dpi, image_format
    def load(self, path):
        if fitz is None: raise RuntimeError("PyMuPDF (fitz) is required to load PDFs")
        p = Path(path)
        if not p.exists(): raise FileNotFoundError(path)
        doc = fitz.open(str(p))
        return {"path": str(p), "document": doc, "page_count": len(doc), "pages": [
            {
                "page_number": i + 1,
                "text": page.get_text("text"),
                "image_count": len(page.get_images(full=True)),
                "drawing_count": len(page.get_drawings()),
            }
            for i, page in enumerate(doc)
        ]}
    def render_page(self, loaded, page_number):
        if fitz is None: raise RuntimeError("PyMuPDF (fitz) is required to render PDFs")
        if page_number < 1 or page_number > len(loaded["document"]): raise IndexError(page_number)
        pix = loaded["document"][page_number - 1].get_pixmap(dpi=self.dpi, alpha=False)
        return pix.tobytes(self.image_format)
