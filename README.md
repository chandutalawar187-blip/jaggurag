# Nexus-RAG

> Intelligent Retrieval-Augmented Generation System

## Overview

Nexus-RAG is a full-stack RAG application that enables intelligent document ingestion, hybrid search, and AI-powered question answering across multiple file formats.

## Tech Stack

- **Backend:** Python, FastAPI, ChromaDB, Gemini/LMStudio
- **Frontend:** React, TypeScript, Vite, Tailwind CSS
- **Infrastructure:** Docker, Docker Compose

## Quick Start

### Local Qwen with Ollama

Nexus uses the local Qwen model by default through Ollama's OpenAI-compatible
API. Install Qwen with `ollama pull qwen2.5:1.5b`; Ollama serves it on port
`11434` and the backend discovers installed models automatically from
`/v1/models`. No API key is required.

For a local backend process, copy `backend/.env.example` to `backend/.env`.
When using Docker Compose, the backend connects to Ollama on the host via
`host.docker.internal:11434`.

```bash
# Clone the repo
git clone <repo-url>
cd nexus-rag

# Start with Docker
docker-compose up --build

# Or run locally
# Backend
cd backend
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload

# Frontend
cd frontend
npm install
cp .env.example .env
npm run dev
```

The local development backend uses port `8000` by default in this workspace
to avoid conflicting with other services on port `8000`. Set `VITE_API_URL`
in `frontend/.env` if your backend uses a different address.

### OCR prerequisites

`Pillow` and `pytesseract` are installed from `backend/requirements.txt`.
`pytesseract` is only a Python wrapper, so local deployments also need the
native Tesseract OCR executable installed and available on `PATH` (or
`TESSERACT_CMD` set in `backend/.env`). The production Docker image installs
both Tesseract and its English language data automatically.

### Handwritten OCR-RAG

The isolated handwritten-document pipeline reuses the secure document upload,
embedding, ChromaDB, and local Qwen services while keeping its own persistent
metadata database and vector collection. It exposes:

```text
POST /api/ocr-rag/documents
GET  /api/ocr-rag/documents/{document_id}
POST /api/ocr-rag/documents/{document_id}/query
```

The default provider is local Tesseract. To use a Qwen-compatible multimodal
OCR endpoint, set `OCR_RAG_PROVIDER=qwen_vision` and
`OCR_RAG_VISION_MODEL=<installed-vision-model>` in `backend/.env`. Qwen vision
results intentionally have no fabricated confidence or bounding-box values.
OCR-RAG metadata is persisted in `backend/data/ocr_rag.db`; original uploads
remain in the dataset storage and are never overwritten.

## Project Structure

```
nexus-rag/
├── backend/          # FastAPI backend + RAG pipeline
├── frontend/         # React + TypeScript frontend
├── docker-compose.yml
└── README.md
```

## License

MIT
