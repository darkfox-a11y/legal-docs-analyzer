# Legal Document Analyzer

Legal Document Analyzer is a `FastAPI` application for uploading legal documents and asking grounded natural-language questions over them using Retrieval-Augmented Generation (RAG). The app extracts text from `PDF` and `DOCX` files, chunks the content, stores embeddings in `Qdrant`, and uses `Gemini` to answer user questions with supporting source excerpts.

## What This Repo Contains

This repository is the application codebase for:

- authentication and user access
- document upload and processing
- text extraction from legal files
- chunking and embedding generation
- vector search and retrieval
- question answering over uploaded documents
- internal evaluation helpers and benchmark scripts

This is primarily the app repo, not the standalone evaluation repo.

## Core Features

- Upload legal documents in `PDF` and `DOCX` format
- Extract text using `PyMuPDF` and `python-docx`
- Chunk long documents with semantic and hierarchical strategies
- Generate embeddings with `SentenceTransformers`
- Store and search vectors in `Qdrant`
- Ask questions against a specific uploaded document
- Return answer text, confidence, and source snippets
- Support benchmark and evaluation workflows for RAG quality analysis

## Tech Stack

- API framework: `FastAPI`
- App server: `Uvicorn`
- Validation/config: `Pydantic`, `pydantic-settings`
- Database: `PostgreSQL`, `SQLAlchemy`
- Vector DB: `Qdrant`
- File parsing: `PyMuPDF`, `python-docx`
- Embeddings: `SentenceTransformers (all-MiniLM-L6-v2)`
- LLMs: `Gemini 2.0 Flash`, `Gemini 2.5 Flash`
- Auth: JWT-based auth
- Testing: `pytest`, `httpx`

## App Architecture

### Backend modules

- `app/auth/`
  - login, signup, JWT, protected routes
- `app/documents/`
  - upload handling, document metadata, text extraction, processing
- `app/rag/`
  - chunking, embeddings, retrieval, vector store, QA, evaluation
- `app/tests/`
  - unit tests and benchmark helper tests

### Storage layers

- `PostgreSQL`
  - users
  - document metadata
  - processing state
- `Qdrant`
  - chunk embeddings
  - chunk text payloads
  - document-filtered retrieval

## RAG Flow

1. User uploads a `PDF` or `DOCX`.
2. The app extracts text and stores metadata.
3. The document is chunked with overlap.
4. Chunks are embedded with `SentenceTransformers`.
5. Embeddings are stored in `Qdrant`.
6. A user question is embedded and matched against stored chunks.
7. Top chunks are sent to `Gemini`.
8. The app returns a grounded answer with source snippets and confidence.

## Main Files

- [main.py](/Users/neerad/legal-doc-analyzer/app/main.py)
- [routes.py](/Users/neerad/legal-doc-analyzer/app/documents/routes.py)
- [processing.py](/Users/neerad/legal-doc-analyzer/app/documents/processing.py)
- [qa.py](/Users/neerad/legal-doc-analyzer/app/rag/qa.py)
- [vector_store.py](/Users/neerad/legal-doc-analyzer/app/rag/vector_store.py)
- [chunking.py](/Users/neerad/legal-doc-analyzer/app/rag/chunking.py)
- [benchmark.py](/Users/neerad/legal-doc-analyzer/app/rag/benchmark.py)

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the API:

```bash
uvicorn app.main:app --reload
```

Run tests:

```bash
pytest
```

## Local Development

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Start the API locally:

```bash
uvicorn app.main:app --reload
```

Common local URLs:

- API root: `http://localhost:8000/`
- Swagger docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- Frontend: `http://localhost:8000/app`

## Environment Variables

Expected `.env` values include:

```env
DATABASE_URL=
GEMINI_API_KEY=
SECRET_KEY=
QDRANT_URL=
QDRANT_API_KEY=
QDRANT_COLLECTION_NAME=legal_documents
```

Optional app/storage settings:

```env
GCP_PROJECT_ID=
GCS_BUCKET_NAME=
GOOGLE_APPLICATION_CREDENTIALS=
```

## Docker

This repo includes a `Dockerfile` and `docker-compose.yml`.

Run with Docker Compose:

```bash
docker-compose up --build
```

The app container exposes:

- `8000:8000`

The container expects the same core env vars as local development:

- `DATABASE_URL`
- `GEMINI_API_KEY`
- `QDRANT_URL`
- `QDRANT_API_KEY`
- `QDRANT_COLLECTION_NAME`
- `SECRET_KEY`

## API Endpoints

### Service endpoints

- `GET /`
  - root metadata
- `GET /health`
  - app health check
- `GET /docs`
  - Swagger UI
- `GET /redoc`
  - ReDoc

### Authentication endpoints

- `POST /auth/register`
  - register a new user
- `POST /auth/login`
  - return bearer token
- `GET /auth/health`
  - auth service health

### Document endpoints

- `POST /documents/upload`
  - upload and process document
- `GET /documents/`
  - list current user documents
- `GET /documents/{document_id}`
  - document details and extracted text
- `DELETE /documents/{document_id}`
  - delete document
- `POST /documents/ask`
  - ask a question against one document
- `GET /documents/{document_id}/stats`
  - document retrieval / chunk stats

## Example API Usage

### Register

```bash
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "password123",
    "name": "Demo User"
  }'
```

### Login

```bash
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "password123"
  }'
```

### Upload a document

```bash
curl -X POST http://localhost:8000/documents/upload \
  -H "Authorization: Bearer <TOKEN>" \
  -F "file=@/absolute/path/to/contract.pdf"
```

### Ask a question

```bash
curl -X POST "http://localhost:8000/documents/ask?question=What%20are%20the%20payment%20terms%3F&document_id=1&top_k=3&detail_level=detailed&model_name=gemini-2.5-flash" \
  -H "Authorization: Bearer <TOKEN>"
```

### Example response

```json
{
  "answer": "The document states that payment is due within 30 days from invoice date.",
  "confidence": "high",
  "detail_level": "detailed",
  "model_name": "gemini-2.5-flash",
  "sources": [
    {
      "chunk_index": 0,
      "text_preview": "Payment is due within 30 days from invoice date...",
      "relevance_score": 0.88
    }
  ]
}
```

## Evaluation Summary

This repo also includes benchmark helpers and scripts used to evaluate the app’s RAG quality.

### Non-judicial public legal/policy benchmark

Archived evaluation showed:

- `100%` query success across `50` queries
- mean latency improved from `3997.62 ms` to `2954.18 ms`
- `P95` latency improved from `6436.56 ms` to `3578.41 ms`
- retrieval depth reduced from `5` chunks to `3`
- high-confidence answers improved from `18%` to `76%`

### Judicial SCOTUS benchmark

Archived evaluation showed:

- average latency around `3.09s`
- composite score improved from `0.4165` to `0.4193`
- judge score improved from `2.58` to `2.74`
- retrieval score stayed at `0.4003`
- groundedness remained low at about `0.21`

These results suggest:

- the app is operationally strong on policy-style legal documents
- judicial QA is harder
- retrieval quality and grounding are still the biggest bottlenecks

## Benchmark Scripts In This Repo

- `scripts/fetch_legal_benchmark_corpus.py`
- `scripts/run_rag_benchmark.py`
- `scripts/generate_ieee_paper_docx.py`

Example benchmark commands:

```bash
python3 scripts/fetch_legal_benchmark_corpus.py
python3 scripts/run_rag_benchmark.py
```

## Frontend

If the `frontend/` directory is present, the app mounts it automatically at:

- `GET /app`

The frontend is intended as a lightweight UI over the backend API for:

- document upload
- model selection
- question submission
- answer and source viewing

The backend API remains the primary product surface in this repo.

## Current Limitations

- no OCR pipeline for scanned PDFs
- judicial groundedness remains low
- general-purpose embedding model may underperform on legal/judicial text
- retrieval depth and chunk sizing still need more ablation

## Resume-Style Summary

Built a Legal Document Analyzer using `FastAPI`, `PostgreSQL`, `Qdrant`, `PyMuPDF`, `python-docx`, `SentenceTransformers`, and `Gemini` to support semantic retrieval and grounded question answering over uploaded legal documents, with benchmarked improvements in latency, confidence, and retrieval workflows across legal-policy and judicial document sets.
