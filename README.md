# SheBots RAG service

This service crawls <https://computer.knu.ac.kr/> (respecting robots.txt and ALLOWLIST) and builds a FAISS index with sentence-transformers embeddings.

Setup

1. Create and activate a Python 3.10+ virtualenv.
2. pip install -r requirements.txt
3. Copy `.env.example` to `.env` and adjust START_URLS or other settings.

Run
   uvicorn rag_main:app --reload --port 8080

Endpoints

- GET /rag/health
- POST /rag/ingest {"full":true}
- GET /rag/search?q=...&k=5
- GET /rag/retrieve?q=...&k=5

Curl examples:
   curl -X POST <http://localhost:8080/rag/ingest> -H "Content-Type: application/json" -d '{"full":true}'
   curl "<http://localhost:8080/rag/search?q=졸업&k=5>"
