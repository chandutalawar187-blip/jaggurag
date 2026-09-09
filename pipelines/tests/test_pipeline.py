import pytest
from pipelines.pdf import PDFLoader
from pipelines.client import parse_json_safely
from pipelines.extractor import DocumentExtractor
from pipelines.schema import PageAnalysis, PageObject, AlphabetItem, Document
from pipelines.retrieval import Retriever
from pipelines.router import QuestionRouter
from pipelines.answers import Calculator, AnswerGenerator

def make_pdf(tmp_path):
    try:
        import fitz
    except ImportError:
        fitz = pytest.importorskip("pymupdf")
    path = tmp_path / "sample.pdf"
    d = fitz.open()
    p = d.new_page(); p.insert_text((72, 72), "Hello document")
    d.save(path); d.close()
    return path

def test_pdf_load_text_and_render(tmp_path):
    loader = PDFLoader(dpi=72)
    loaded = loader.load(make_pdf(tmp_path))
    assert "Hello document" in loaded["pages"][0]["text"]
    assert loader.render_page(loaded, 1).startswith(b"\x89PNG")

def test_malformed_json_is_safe():
    assert parse_json_safely("```json\n{\"objects\": []}\n```") == {"objects": []}
    assert parse_json_safely("not json") == {}

def test_visual_extraction_and_mock_client(tmp_path):
    class C:
        def analyze(self, image, prompt): return '{"objects":[{"label":"cat"}]}'
    try:
        import fitz
    except ImportError:
        fitz = pytest.importorskip("pymupdf")
    path = tmp_path / "blank.pdf"
    d = fitz.open(); d.new_page(); d.save(path); d.close()
    loader = PDFLoader()
    doc = DocumentExtractor(C(), loader).extract(loader.load(path))
    assert doc.pages[0].objects[0].label == "cat"

def test_retrieval_and_routing():
    d = Document(source="x", pages=[PageAnalysis(page_number=2, text="A lesson")])
    r = Retriever(d)
    assert r.search("lesson")[0]["page"] == 2
    router = QuestionRouter()
    assert router.route("How many alphabet words?") == "alphabet_count"
    assert router.route("What objects are on page 2?") == "page_objects"
    assert router.route("Why does this work?") == "reasoning"

def test_deterministic_answers_and_missing_info():
    p = PageAnalysis(page_number=1, objects=[PageObject(label="tree")],
                     alphabet_items=[AlphabetItem(letter="A", word="apple")])
    d = Document(source="x", pages=[p]); r = Retriever(d)
    g = AnswerGenerator()
    assert g.answer("how many alphabet words", "alphabet_count", d, r) == "1"
    assert g.answer("list alphabet", "alphabet_list", d, r) == "A: apple"
    assert g.answer("objects on page 1", "page_objects", d, r) == "tree"
    assert "couldn't" in g.answer("what is the weather?", "generic", d, r).lower()
    assert Calculator.word_count("One, two three!") == 3
