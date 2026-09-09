# Isolated document-understanding pipeline

This directory is a standalone, opt-in PDF understanding pipeline. It is not
imported by the existing backend or frontend and does not change any current
API route or application behavior.

## Architecture

`DocumentPipeline` performs the following stages:

1. `PDFLoader` opens a PDF with PyMuPDF, records page count and text, and
   renders individual pages only when visual analysis is needed.
2. `DocumentExtractor` creates typed page records. It preserves extracted text
   and page numbers, derives simple alphabet items from text, and sends
   image-heavy pages to the injected Qwen vision client.
3. `Retriever` performs lightweight page-preserving term retrieval. It is
   intentionally replaceable later with a vector index.
4. `QuestionRouter` classifies list/count/page-object/reasoning questions.
5. `AnswerGenerator` performs exact counts and list answers in Python. Qwen is
   used only for grounded reasoning over retrieved context.

The structured representation contains pages, headings, objects, image labels,
tables, lists, relationships, and alphabet items. Unknown values remain empty
or `null`; malformed provider JSON is ignored safely.

## Installation

The existing backend requirements already include PyMuPDF, Pydantic, OpenAI,
httpx, and pytest. No existing dependency manifest was changed. If using a
separate environment, install only the optional pipeline dependencies:

```text
pip install pymupdf pydantic openai httpx pytest
```

## Qwen configuration

The provider adapter is isolated in `pipelines/client.py`. It supports
OpenAI-compatible Qwen endpoints and injected mock responders:

```text
QWEN_API_KEY=...
QWEN_MODEL=qwen-vl-plus
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
PIPELINE_RENDER_DPI=144
PIPELINE_IMAGE_FORMAT=png
PIPELINE_MAX_PAGES=100
```

No key is needed for tests. Pass a client with an `analyze(image, prompt)`
callable to `DocumentPipeline`.

## Python API

```python
from pipelines import DocumentPipeline

pipeline = DocumentPipeline()
document = pipeline.process_document("document.pdf")
print(pipeline.get_page(1))
print(pipeline.ask("What is A for?"))
```

The pipeline must be processed before `ask()` or `get_page()` is used.
Deterministic questions such as “How many words start with A?” do not ask the
LLM to count.

## Employee face recognition (opt-in)

`EmployeeFaceRegistry` provides an isolated enrollment and recognition flow.
Enrollment requires explicit consent and one detectable face per image. It
stores only the averaged face embedding and employee ID/name in SQLite; source
photographs are read but never copied into the registry. Recognition returns
only matches below the configured distance threshold, so an unknown face is
not assigned an employee.

The default provider is optional and loaded only when used:

```text
pip install face-recognition
```

```python
from pipelines import EmployeeFaceRegistry

registry = EmployeeFaceRegistry("data/employee_faces.sqlite3")
registry.enroll("emp-42", "Asha Rao", ["asha-1.jpg", "asha-2.jpg"], consent=True)
matches = registry.recognize("camera-frame.jpg")
for match in matches:
    print(match.employee_id, match.name, match.confidence)
```

For tests or another detector, inject an object implementing
`extract(image_bytes) -> list[DetectedFace]`. Keep the database protected with
the same access controls as other biometric data, and provide deletion and
manual verification workflows before using a match for an employment decision.
The web application's `/people` API and People screen use this same registry.
The default detector is optional; install `face-recognition` in the backend
environment (and ensure the `pipelines` package is on `PYTHONPATH`) before
enrolling or recognizing real images.

## CLI

```bash
python -m pipelines document.pdf
```

The command loads the document and accepts questions until `exit`, `quit`, or
EOF. Vision pages require Qwen configuration; text-only documents can still be
loaded with an injected client in Python.

## Tests

```bash
python -m pytest pipelines/tests -q
```

Tests use temporary PDFs and mock Qwen responses. They never start the
existing FastAPI application and never call an external model.

## Future integration

Integration is deliberately not included. A future adapter can convert
`Document`, `PageAnalysis`, and answer/source data into the existing
application's types without importing the existing RAG implementation into
this package.
