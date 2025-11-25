# rag_main.py — HYBRID RAG (Semantic + Keyword + URL Expansion + Chunk Limit)

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import os, json, re
from dotenv import load_dotenv
load_dotenv()

import sys
from pathlib import Path
import logging

pkg_root = Path(__file__).resolve().parents[1]
if str(pkg_root) not in sys.path:
    sys.path.insert(0, str(pkg_root))

from rag.ingest import ingest
from rag.store import FaissStore
from rag.embeddings import get_embedding_model

app = FastAPI(title='SheBots Hybrid RAG')

# ----------------------------- ENV -----------------------------
INDEX_PATH = os.getenv('INDEX_PATH', '.data/test/faiss_index')
DOCSTORE_PATH = os.getenv('DOCSTORE_PATH', '.data/test/docstore.jsonl')
START_URLS = os.getenv('START_URLS', 'https://computer.knu.ac.kr/eng/').split(';')
ALLOWLIST = os.getenv('ALLOWLIST', 'https://computer.knu.ac.kr/').split(';')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
TOP_K = int(os.getenv('TOP_K', '5'))
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "8"))   # ⭐ NEW: Limit final context

# ----------------------------- Logging -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger("RAG")

# ----------------------------- Load Embedding Model -----------------------------
logger.info("Loading embedding model globally...")
embedder = get_embedding_model(EMBEDDING_MODEL)

# ----------------------------- Utils -----------------------------
def load_docstore():
    docs = []
    if os.path.exists(DOCSTORE_PATH):
        with open(DOCSTORE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    docs.append(json.loads(line))
                except:
                    pass
    return docs


def keyword_match(query: str, docs: list):
    q = query.lower()
    results = []
    for d in docs:
        text = d.get("text") or d.get("content") or ""
        if q in text.lower():
            results.append(d)
    return results


def group_by_url(docs: list):
    grouped = {}
    for d in docs:
        url = d.get("url", "unknown")
        grouped.setdefault(url, []).append(d)
    return grouped


def merge_unique(list1, list2):
    seen = set()
    merged = []
    for d in list1 + list2:
        key = (d.get("url"), d.get("text"))
        if key not in seen:
            merged.append(d)
            seen.add(key)
    return merged


def build_store(dim: int) -> FaissStore:
    store = FaissStore(dim, INDEX_PATH, DOCSTORE_PATH)
    store.load_or_create()
    return store


# ----------------------------- Health -----------------------------
@app.get("/rag/health")
def rag_health():
    total = 0
    if os.path.exists(DOCSTORE_PATH):
        with open(DOCSTORE_PATH, "r", encoding="utf-8") as f:
            total = sum(1 for _ in f)
    return {"ok": True, "documents": total}


# ----------------------------- Ingest -----------------------------
class IngestRequest(BaseModel):
    full: bool | None = False

@app.post("/rag/ingest")
def rag_ingest(req: IngestRequest):
    stats = ingest(
        START_URLS, ALLOWLIST, INDEX_PATH, DOCSTORE_PATH, EMBEDDING_MODEL,
        max_pages=int(os.getenv("MAX_PAGES", "300")),
        max_depth=int(os.getenv("MAX_DEPTH", "2")),
        delay_ms=int(os.getenv("CRAWL_DELAY_MS", "1500")),
    )
    return stats


# ----------------------------- Hybrid Search -----------------------------
@app.get("/rag/search")
def rag_search(q: str = Query(..., alias="query"), k: int = TOP_K):

    if not q:
        raise HTTPException(400, "Query required")

    emb = embedder.encode([q])[0]
    store = build_store(len(emb))

    # 1) Semantic
    semantic = store.search(emb, k=k)

    # 2) Keyword
    docs_all = load_docstore()
    key_matches = keyword_match(q, docs_all)

    # 3) Merge
    merged = merge_unique(semantic, key_matches)

    # 4) Expand to all chunks of these URLs
    grouped = group_by_url(docs_all)
    expanded = []
    for url in {d.get("url") for d in merged}:
        expanded.extend(grouped.get(url, []))

    # 5) ⭐ LIMIT CONTEXT to MAX_CHUNKS
    if len(expanded) > MAX_CHUNKS:
        expanded = expanded[:MAX_CHUNKS]

    return {
        "query": q,
        "results": expanded,
        "semantic_count": len(semantic),
        "keyword_count": len(key_matches),
        "final_chunks": len(expanded)
    }


# POST version
class SearchBody(BaseModel):
    query: str
    k: int = TOP_K

@app.post("/rag/search")
def rag_search_post(body: SearchBody):
    return rag_search(body.query, body.k)


# ----------------------------- Retrieve -----------------------------
@app.get("/rag/retrieve")
def rag_retrieve(q: str, k: int = TOP_K):
    data = rag_search(q, k)
    return {"query": q, "results": data["results"]}


# ----------------------------- Run -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8001")))
