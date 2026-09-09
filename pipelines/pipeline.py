from .config import Settings
from .pdf import PDFLoader
from .client import QwenVisionClient
from .extractor import DocumentExtractor
from .retrieval import Retriever
from .router import QuestionRouter
from .answers import AnswerGenerator

class DocumentPipeline:
    def __init__(self, settings=None, client=None):
        self.settings = settings or Settings.from_env()
        self.loader = PDFLoader(self.settings.dpi, self.settings.image_format)
        self.client = client or QwenVisionClient(self.settings.api_key, self.settings.model, self.settings.base_url)
        self.router, self.document, self.retriever = QuestionRouter(), None, None
    def process_document(self, path):
        self.document = DocumentExtractor(self.client, self.loader).extract(self.loader.load(path))
        self.retriever = Retriever(self.document)
        return self.document
    def get_document(self): return self.document
    def get_page(self, page_number): return self.retriever.page(page_number) if self.retriever else None
    def ask(self, question):
        if not self.document: raise RuntimeError("Process a document before asking questions")
        route = self.router.route(question)
        return AnswerGenerator(self.client).answer(question, route, self.document, self.retriever)
