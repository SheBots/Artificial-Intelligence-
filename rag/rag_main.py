# rag_main.py — Universal Multi-Stage HYBRID RAG
# - 100% domain-agnostic / no hard-coded synonyms
# - Semantic search (FAISS)
# - Keyword relevance scoring (query-token overlap)
# - Score fusion + reranking
# - Adaptive chunk limit
# - Works for ANY question or dataset
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import os
import json
import re
from dotenv import load_dotenv

load_dotenv()

import logging

# Remove manual sys.path manipulation and import via package
from .ingest import ingest
from .store import FaissStore
from .embeddings import get_embedding_model

app = FastAPI(title="Universal Hybrid RAG")

# ----------------------------- ENV -----------------------------
# Fix default paths to ./data instead of .data
INDEX_PATH = os.getenv("INDEX_PATH", "./data/test/faiss_index")
DOCSTORE_PATH = os.getenv("DOCSTORE_PATH", "./data/test/docstore.jsonl")
START_URLS = os.getenv("START_URLS", "").split(";")
ALLOWLIST = os.getenv("ALLOWLIST", "").split(";")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

TOP_K = int(os.getenv("TOP_K", "5"))
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "8"))
SEMANTIC_CAND_MULTIPLIER = int(os.getenv("SEMANTIC_CAND_MULTIPLIER", "4"))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
)
logger = logging.getLogger("RAG")

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


def tokenize(text: str):
    """Universal tokenizer: handles English, Korean, numbers, etc."""
    return re.findall(r"[0-9A-Za-z가-힣]+", text.lower())


def keyword_rank(query: str, docs: list, max_results: int = 30):
    """
    Universal keyword relevance:
    - token overlaps
    - no domain-specific synonym injection
    - scoring = occurrences in title * 2 + occurrences in text
    """
    q_tokens = tokenize(query)

    if not q_tokens:
        return []

    scored_docs = []

    for d in docs:
        text = (d.get("text") or d.get("content") or "").lower()
        title = (d.get("title") or "").lower()

        if not text and not title:
            continue

        score = 0.0
        for t in q_tokens:
            if len(t) < 2:
                continue

            score += 2.0 * title.count(t)
            score += 1.0 * text.count(t)

        if score > 0:
            newd = dict(d)
            newd["keyword_score"] = float(score)
            scored_docs.append(newd)

    scored_docs.sort(key=lambda x: x.get("keyword_score", 0), reverse=True)
    return scored_docs[:max_results]


def build_store(dim: int) -> FaissStore:
    store = FaissStore(dim, INDEX_PATH, DOCSTORE_PATH)
    store.load_or_create()
    return store


def merge(semantic, keyword):
    """Merge semantic + keyword candidates without domain bias."""
    merged = {}
    for d in semantic:
        key = (d.get("url"), d.get("text"))
        merged[key] = {
            **d,
            "semantic_score": float(d.get("score", 0.0)),
            "keyword_score": float(d.get("keyword_score", 0.0)),
        }

    for d in keyword:
        key = (d.get("url"), d.get("text"))
        if key not in merged:
            merged[key] = {
                **d,
                "semantic_score": float(d.get("semantic_score", 0.0)),
                "keyword_score": float(d.get("keyword_score", 0.0)),
            }
        else:
            merged[key]["keyword_score"] = max(
                merged[key]["keyword_score"],
                float(d.get("keyword_score", 0.0)),
            )

    return list(merged.values())


def rerank(cands: list, k: int, max_chunks: int):
    """
    Final universal fusion score:
        final = 0.7 * semantic + 0.3 * keyword_norm
    No boosts or domain conditions.
    """
    if not cands:
        return []

    max_key = max((c.get("keyword_score", 0.0) for c in cands), default=1.0)
    if max_key <= 0:
        max_key = 1.0

    final = []
    for c in cands:
        sem = float(c.get("semantic_score", 0.0))
        key = float(c.get("keyword_score", 0.0))
        key_norm = key / max_key

        score = 0.7 * sem + 0.3 * key_norm

        newc = dict(c)
        newc["final_score"] = float(score)
        final.append(newc)

    final.sort(key=lambda x: x["final_score"], reverse=True)

    limit = min(len(final), max_chunks, max(k, 1))
    return final[:limit]

# ----------------------------- Health -----------------------------

@app.get("/rag/health")
def rag_health():
    count = 0
    if os.path.exists(DOCSTORE_PATH):
        with open(DOCSTORE_PATH, "r", encoding="utf-8") as f:
            count = sum(1 for _ in f)
    return {"ok": True, "documents": count}

# ----------------------------- Ingest -----------------------------

class IngestRequest(BaseModel):
    full: bool | None = False

@app.post("/rag/ingest")
def rag_ingest(req: IngestRequest):
    stats = ingest(
        START_URLS,
        ALLOWLIST,
        INDEX_PATH,
        DOCSTORE_PATH,
        EMBEDDING_MODEL,
        max_pages=int(os.getenv("MAX_PAGES", "300")),
        max_depth=int(os.getenv("MAX_DEPTH", "2")),
        delay_ms=int(os.getenv("CRAWL_DELAY_MS", "1500")),
    )
    return stats


# ----------------------------- Search -----------------------------

@app.get("/rag/search")
def rag_search(query: str = Query(...), k: int = TOP_K):

    if not query:
        raise HTTPException(400, "Query required")

    # 1) Semantic search
    vector = embedder.encode([query])[0]
    store = build_store(len(vector))

    sem_k = max(k * SEMANTIC_CAND_MULTIPLIER, k, 8)
    semantic_raw = store.search(vector, k=sem_k)
    for r in semantic_raw:
        r["semantic_score"] = float(r.get("score", 0.0))

    # 2) Keyword search
    docs_all = load_docstore()
    keyword_raw = keyword_rank(query, docs_all)

    # 3) Merge + rerank
    merged = merge(semantic_raw, keyword_raw)
    final = rerank(merged, k, MAX_CHUNKS)

    return {
        "query": query,
        "results": final,
        "semantic_count": len(semantic_raw),
        "keyword_count": len(keyword_raw),
        "final_chunks": len(final),
    }


# POST version kept for compatibility
class SearchBody(BaseModel):
    query: str
    k: int = TOP_K

@app.post("/rag/search")
def rag_search_post(body: SearchBody):
    return rag_search(body.query, body.k)


# ----------------------------- Retrieve (shortcut) -----------------------------

@app.get("/rag/retrieve")
def rag_retrieve(q: str, k: int = TOP_K):
    return rag_search(q, k)


# ----------------------------- Run -----------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8001")),
        reload=True,
    )