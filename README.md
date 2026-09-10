# Nexus-RAG

> Intelligent Retrieval-Augmented Generation System

## Overview

Nexus-RAG is a full-stack RAG application for document ingestion, hybrid search, and AI-powered question answering.

## Tech Stack

- Backend: Python, FastAPI, ChromaDB, Gemini/LMStudio
- Frontend: React, TypeScript, Vite, Tailwind CSS
- Infrastructure: Docker, Docker Compose

## Quick Start

### Local Qwen with Ollama

Nexus uses the local Qwen model by default through Ollama's OpenAI-compatible API. Install Qwen with `ollama pull qwen2.5:1.5b`; Ollama serves it on port `11434`, and the backend discovers installed models automatically from `/v1/models`.

For a local backend process, copy `backend/.env.example` to `backend/.env`. When using Docker Compose, the backend connects to Ollama on the host via `host.docker.internal:11434`.

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

The local development backend uses port `8000` by default in this workspace to avoid conflicts with other services. Set `VITE_API_URL` in `frontend/.env` if your backend uses a different address.

## Project Structure

```text
nexus-rag/
├── backend/          # FastAPI backend + RAG pipeline
├── frontend/         # React + TypeScript frontend
├── docker-compose.yml
├── README.md
└── .gitignore
```

## License

MIT
