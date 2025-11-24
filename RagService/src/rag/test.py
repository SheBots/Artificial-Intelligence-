import os
import sys
from urllib.parse import urlparse
from pathlib import Path

# Ensure package root is available
pkg_root = Path(__file__).resolve().parents[1]
if str(pkg_root) not in sys.path:
    sys.path.insert(0, str(pkg_root))

from rag.ingest import ingest
from rag.store import FaissStore
from rag.embeddings import get_embedding_model

# Hardcoded test parameters (run with: python test.py)
DEFAULT_URL = "https://cse.knu.ac.kr/index.php"
DEFAULT_QUERY = "현장실습"
DEFAULT_MAX_PAGES = 2
DEFAULT_MAX_DEPTH = 0
DEFAULT_K = 3
DEFAULT_SHOW_DOCS = 3
DEFAULT_INDEX_PATH = "./data/test/faiss_index"
DEFAULT_DOCSTORE_PATH = "./data/test/docstore.jsonl"
DEFAULT_DELAY_MS = 1000
DEFAULT_EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')


def derive_allowlist(url: str) -> str:
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}/"


def load_store(index_path: str, docstore_path: str, dim: int) -> FaissStore:
    store = FaissStore(dim, index_path, docstore_path)
    store.load_or_create()
    return store


def print_docstore(docstore_path: str, limit: int):
    if not os.path.exists(docstore_path):
        print(f"Docstore not found at {docstore_path}")
        return
    print(f"\n=== Stored Documents (showing first {limit} chunks) ===")
    shown = 0
    import json
    with open(docstore_path, 'r', encoding='utf-8') as f:
        for line in f:
            if shown >= limit:
                break
            try:
                d = json.loads(line)
                text_preview_full = d.get('text', '')
                text_preview = (text_preview_full[:300] + '...') if len(text_preview_full) > 300 else text_preview_full
                print(f"[{shown+1}] URL: {d.get('url','')} | Title: {d.get('title','')}")
                print(f"    Chunk ID: {d.get('chunk_id','')} | Fetched: {d.get('fetched_at','')}")
                print(f"    Text: {text_preview}\n")
            except Exception as e:
                print(f"Failed to parse line: {e}")
            shown += 1


def run_default():
    start_url = DEFAULT_URL
    allowlist = [derive_allowlist(start_url)]
    start_urls = [start_url]

    print("=== Ingestion Phase (hardcoded defaults) ===")
    stats = ingest(
        start_urls,
        allowlist,
        index_path=DEFAULT_INDEX_PATH,
        docstore_path=DEFAULT_DOCSTORE_PATH,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        max_pages=DEFAULT_MAX_PAGES,
        max_depth=DEFAULT_MAX_DEPTH,
        delay_ms=DEFAULT_DELAY_MS
    )
    print("Ingestion Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print_docstore(DEFAULT_DOCSTORE_PATH, limit=DEFAULT_SHOW_DOCS)

    print("=== Retrieval Phase (hardcoded query) ===")
    model = get_embedding_model(DEFAULT_EMBEDDING_MODEL)
    emb = model.encode([DEFAULT_QUERY])[0]
    store = load_store(DEFAULT_INDEX_PATH, DEFAULT_DOCSTORE_PATH, len(emb))
    results = store.search(emb, k=DEFAULT_K)
    if not results:
        print("No retrieval results.")
    else:
        print(f"Top {len(results)} results for query: '{DEFAULT_QUERY}'\n")
        for i, r in enumerate(results, start=1):
            text_preview_full = r.get('text','')
            text_preview = (text_preview_full[:300] + '...') if len(text_preview_full) > 300 else text_preview_full
            print(f"[{i}] score={r.get('score',0):.4f}")
            print(f"    URL: {r.get('url','')} | Title: {r.get('title','')}")
            print(f"    Text: {text_preview}\n")

    print("Done.")


if __name__ == '__main__':
    run_default()
