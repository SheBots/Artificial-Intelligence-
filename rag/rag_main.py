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
from pathlib import Path

# Load .env next to this file to be robust to CWD
load_dotenv(Path(__file__).with_name(".env"))

import logging

# Remove manual sys.path manipulation and import via package
from .ingest import ingest
from .store import FaissStore
from .embeddings import get_embedding_model

app = FastAPI(title="Universal Hybrid RAG")

# ----------------------------- ENV -----------------------------
# Stable defaults anchored to repo root, robust across CWDs
PKG_DIR = Path(__file__).resolve().parent
ROOT_DIR = PKG_DIR.parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data" / "test"))

INDEX_PATH = os.getenv("INDEX_PATH", str(DATA_DIR / "faiss_index"))
DOCSTORE_PATH = os.getenv("DOCSTORE_PATH", str(DATA_DIR / "docstore.jsonl"))
START_URLS = os.getenv("START_URLS", "").split(";") if os.getenv("START_URLS") else []
ALLOWLIST = os.getenv("ALLOWLIST", "").split(";") if os.getenv("ALLOWLIST") else []
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

TOP_K = int(os.getenv("TOP_K", "5"))
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "12"))
SEMANTIC_CAND_MULTIPLIER = int(os.getenv("SEMANTIC_CAND_MULTIPLIER", "6"))

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


MAJOR_ALIASES = {
    "platform_software": {"platform software", "플랫폼소프트웨어"},
    "global_software": {"global software", "글솝", "glassop", "global sw"},
    "data_science": {"data science", "데이터과학"},
    "advanced_computing": {"advanced computing", "심화컴퓨터공학", "abeek"},
}

def detect_target_major(query: str) -> str | None:
    q = (query or "").lower()
    for key, aliases in MAJOR_ALIASES.items():
        if any(a in q for a in aliases):
            return key
    return None

def keyword_rank(query: str, docs: list, max_results: int = 50):
    """
    Enhanced keyword relevance for requirements & numeric facts:
    - token overlaps
    - title boosts (2x)
    - extra boosts for digits and requirement terms
    """
    q_tokens = tokenize(query)
    if not q_tokens:
        return []

    numeric_terms = set([t for t in q_tokens if t.isdigit()])
    req_terms = {"credit", "credits", "학점", "internship", "인턴", "요건", "requirements", "졸업", "필수"}
    target_major = detect_target_major(query)

    scored_docs = []
    for d in docs:
        text = (d.get("text") or d.get("content") or "").lower()
        title = (d.get("title") or "").lower()
        url = (d.get("url") or "").lower()
        if not text and not title:
            continue

        score = 0.0
        for t in q_tokens:
            if len(t) < 2:
                continue
            title_hits = title.count(t)
            text_hits = text.count(t)
            base = 2.0 * title_hits + 1.0 * text_hits
            if t in numeric_terms:
                base *= 1.6
            if t in req_terms:
                base *= 1.4
            score += base

        # Major-aware boosting/penalty based on title/url path
        if target_major:
            target_aliases = MAJOR_ALIASES.get(target_major, set())
            # Found target major in title or url/path
            if any(a in title or a in url for a in target_aliases):
                score *= 1.6
            else:
                # Penalize obvious mismatches (aliases from other majors)
                other_aliases = set().union(*(v for k, v in MAJOR_ALIASES.items() if k != target_major))
                if any(a in title or a in url for a in other_aliases):
                    score *= 0.7

        if score > 0:
            newd = dict(d)
            newd["keyword_score"] = float(score)
            scored_docs.append(newd)

    scored_docs.sort(key=lambda x: x.get("keyword_score", 0.0), reverse=True)
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
    Fusion score tuned for numeric requirement queries:
        final = 0.55 * semantic + 0.45 * keyword_norm
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

        score = 0.55 * sem + 0.45 * key_norm

        newc = dict(c)
        newc["final_score"] = float(score)
        final.append(newc)

    final.sort(key=lambda x: x["final_score"], reverse=True)

    # Group by document (url) and keep best per doc to avoid one doc dominating
    by_doc: dict[str, dict] = {}
    for item in final:
        doc_key = str(item.get("url"))
        prev = by_doc.get(doc_key)
        if (prev is None) or (item.get("final_score", 0.0) > prev.get("final_score", 0.0)):
            by_doc[doc_key] = item

    grouped = list(by_doc.values())
    grouped.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)

    limit = min(len(grouped), max_chunks, max(k, 1))
    return grouped[:limit]

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