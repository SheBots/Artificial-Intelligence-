# python RagService\src\rag\rag_main.py
import os
import sys
import json
import argparse
from urllib.parse import urlparse
from pathlib import Path

pkg_root = Path(__file__).resolve().parents[1]
if str(pkg_root) not in sys.path:
    sys.path.insert(0, str(pkg_root))

from rag.ingest import ingest
from rag.store import FaissStore
from rag.embeddings import get_embedding_model

DEFAULT_URL = "https://cse.knu.ac.kr/index.php"
DEFAULT_QUERY = "현장실습"
DEFAULT_MAX_PAGES = 2
DEFAULT_MAX_DEPTH = 0
DEFAULT_K = int(os.getenv('TOP_K', '5'))
DEFAULT_INDEX_PATH = os.getenv('INDEX_PATH', "./data/test/faiss_index")
DEFAULT_DOCSTORE_PATH = os.getenv('DOCSTORE_PATH', "./data/test/docstore.jsonl")
DEFAULT_DELAY_MS = int(os.getenv('CRAWL_DELAY_MS', '1000'))
DEFAULT_EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')


def derive_allowlist(url: str) -> str:
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}/"


def load_store(index_path: str, docstore_path: str, dim: int) -> FaissStore:
    store = FaissStore(dim, index_path, docstore_path)
    store.load_or_create()
    return store


def cmd_ingest(args):
    start_url = args.url or DEFAULT_URL
    start_urls = [start_url]
    allowlist = [derive_allowlist(start_url)]
    stats = ingest(
        start_urls,
        allowlist,
        index_path=args.index_path,
        docstore_path=args.docstore_path,
        embedding_model=args.model,
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        delay_ms=args.delay_ms
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def _count_docstore(docstore_path: str) -> int:
    if not os.path.exists(docstore_path):
        return 0
    total = 0
    with open(docstore_path, 'r', encoding='utf-8') as f:
        for _ in f:
            total += 1
    return total


def cmd_health(args):
    total = _count_docstore(args.docstore_path)
    print(json.dumps({"ok": True, "documents": total}, ensure_ascii=False, indent=2))


def _search_common(query: str, k: int, index_path: str, docstore_path: str, model_name: str):
    if not query:
        raise SystemExit("Query string required.")
    model = get_embedding_model(model_name)
    emb = model.encode([query])[0]
    store = load_store(index_path, docstore_path, len(emb))
    results = store.search(emb, k=k)
    return results


def cmd_search(args):
    results = _search_common(args.query, args.k, args.index_path, args.docstore_path, args.model)
    resp = {
        "query": args.query,
        "results": results
    }
    print(json.dumps(resp, ensure_ascii=False, indent=2))


def cmd_retrieve(args):
    results = _search_common(args.query, args.k, args.index_path, args.docstore_path, args.model)
    items = [
        {
            "url": r.get("url", ""),
            "title": r.get("title", ""),
            "text": r.get("text", ""),
            "score": r.get("score", 0)
        } for r in results
    ]
    resp = {
        "query": args.query,
        "results": items
    }
    print(json.dumps(resp, ensure_ascii=False, indent=2))


def build_parser():
    p = argparse.ArgumentParser(description="Local RAG test tool (mirrors API outputs).")
    sub = p.add_subparsers(dest="command", required=True)

    # ingest
    sp_ingest = sub.add_parser("ingest", help="Run ingestion (like /rag/ingest).")
    sp_ingest.add_argument("--url", default=DEFAULT_URL)
    sp_ingest.add_argument("--index-path", default=DEFAULT_INDEX_PATH)
    sp_ingest.add_argument("--docstore-path", default=DEFAULT_DOCSTORE_PATH)
    sp_ingest.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    sp_ingest.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES)
    sp_ingest.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    sp_ingest.add_argument("--delay-ms", type=int, default=DEFAULT_DELAY_MS)
    sp_ingest.set_defaults(func=cmd_ingest)

    # health
    sp_health = sub.add_parser("health", help="Show docstore count (like /rag/health).")
    sp_health.add_argument("--docstore-path", default=DEFAULT_DOCSTORE_PATH)
    sp_health.set_defaults(func=cmd_health)

    # search
    sp_search = sub.add_parser("search", help="Semantic search (like /rag/search).")
    sp_search.add_argument("--query", required=True)
    sp_search.add_argument("-k", type=int, default=DEFAULT_K)
    sp_search.add_argument("--index-path", default=DEFAULT_INDEX_PATH)
    sp_search.add_argument("--docstore-path", default=DEFAULT_DOCSTORE_PATH)
    sp_search.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    sp_search.set_defaults(func=cmd_search)

    # retrieve
    sp_retrieve = sub.add_parser("retrieve", help="Minimal retrieval (like /rag/retrieve).")
    sp_retrieve.add_argument("--query", required=True)
    sp_retrieve.add_argument("-k", type=int, default=DEFAULT_K)
    sp_retrieve.add_argument("--index-path", default=DEFAULT_INDEX_PATH)
    sp_retrieve.add_argument("--docstore-path", default=DEFAULT_DOCSTORE_PATH)
    sp_retrieve.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    sp_retrieve.set_defaults(func=cmd_retrieve)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

# python RagService\src\rag\test.py ingest
# python RagService\src\rag\test.py search --query "현장실습" -k 5
# python RagService\src\rag\test.py retrieve --query "현장실습" -k 5
# python RagService\src\rag\test.py health