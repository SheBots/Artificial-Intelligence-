from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import os, json
from dotenv import load_dotenv
load_dotenv()

import sys
from pathlib import Path
import logging
# When executing this file directly (python src/rag/rag_main.py), ensure the
# package root (RagService/src) is on sys.path so package imports work.
pkg_root = Path(__file__).resolve().parents[1]
if str(pkg_root) not in sys.path:
    sys.path.insert(0, str(pkg_root))

from rag.ingest import ingest
from rag.store import FaissStore
from rag.embeddings import get_embedding_model, embed_texts

app = FastAPI(title='SheBots RAG')

INDEX_PATH = os.getenv('INDEX_PATH','./data/faiss_index')
DOCSTORE_PATH = os.getenv('DOCSTORE_PATH','./data/docstore.jsonl')
START_URLS = os.getenv('START_URLS','https://computer.knu.ac.kr/eng/;https://computer.knu.ac.kr/board/').split(';')
ALLOWLIST = os.getenv('ALLOWLIST','https://computer.knu.ac.kr/').split(';')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL','sentence-transformers/all-MiniLM-L6-v2')
TOP_K = int(os.getenv('TOP_K','5'))

# Logging / debug toggle
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=LOG_LEVEL, format='[%(asctime)s] %(levelname)s %(name)s - %(message)s')
logger = logging.getLogger('SheBotsRAG')
RAG_DEBUG = os.getenv('RAG_DEBUG', '0').lower() in ('1','true','yes','on')

def _log_retrieval(query: str, results: list, k: int):
    try:
        logger.info(f"[RAG] query: {query} (k={k}, hits={len(results)})")
        for i, r in enumerate(results[:k], start=1):
            score = r.get('score', 0)
            title = (r.get('title','') or '')[:120]
            url = r.get('url','')
            text = (r.get('text','') or '').replace('\n',' ')[:500]
            logger.info(f"[RAG] {i}: score={score} url={url} title={title}")
            logger.debug(f"[RAG] {i} text: {text}")
    except Exception:
        logger.exception("Failed to log RAG results")

class IngestRequest(BaseModel):
    full: bool | None = False


@app.get('/rag/health')
def health():
    # count docs
    total = 0
    if os.path.exists(DOCSTORE_PATH):
        with open(DOCSTORE_PATH,'r',encoding='utf-8') as f:
            for _ in f:
                total += 1
    return {'ok': True, 'documents': total}


@app.post('/rag/ingest')
def rag_ingest(req: IngestRequest):
    res = ingest(START_URLS, ALLOWLIST, INDEX_PATH, DOCSTORE_PATH, EMBEDDING_MODEL, max_pages=int(os.getenv('MAX_PAGES','300')), max_depth=int(os.getenv('MAX_DEPTH','2')), delay_ms=int(os.getenv('CRAWL_DELAY_MS','1500')))
    return res


@app.get('/rag/search')
def rag_search(
    q: str = Query(..., alias='query'),
    k: int = TOP_K,
    debug: bool = False
):
    if not q:
        raise HTTPException(status_code=400, detail='q is required')
    model = get_embedding_model()
    emb = model.encode([q])[0]
    store = FaissStore(len(emb), INDEX_PATH, DOCSTORE_PATH)
    store.load_or_create()
    results = store.search(emb, k=k)
    if RAG_DEBUG or debug:
        _log_retrieval(q, results, k)
    return {'query': q, 'results': results}


@app.get('/rag/retrieve')
def rag_retrieve(q: str, k: int = TOP_K, debug: bool = False):
    # same as search but minimal fields
    res = rag_search(q, k, debug)
    items = [{'url': r['url'], 'title': r.get('title',''), 'text': r.get('text',''), 'score': r.get('score',0)} for r in res['results']]
    return {'query': q, 'results': items}


if __name__ == '__main__':
    # Allow running the module directly: python RagService\rag_main.py
    # This will start uvicorn programmatically on port 8001 (change via env var PORT)
    import uvicorn
    port = int(os.getenv('PORT', '8001'))
    host = os.getenv('HOST', '127.0.0.1')
    print(f"Starting SheBots RAG on http://{host}:{port} (CTRL+C to stop)")
    # Pass the app object directly to avoid uvicorn trying to import the module
    uvicorn.run(app, host=host, port=port, reload=False)