import os, time, json
from .loader import crawl
from .clean import clean_text
from .splitter import split_text
from .embeddings import embed_texts
from .store import FaissStore, Doc

def ingest(start_urls, allowlist, index_path, docstore_path, embedding_model, max_pages=300, max_depth=2, delay_ms=1500):
    pages = crawl(start_urls, allowlist, max_pages=max_pages, max_depth=max_depth, delay_ms=delay_ms)
    store = None
    all_chunks = []
    for p in pages:
        url = p['url']
        title = p.get('title','')
        text = clean_text(p['text'])
        chunks = split_text(text)
        for i, c in enumerate(chunks):
            meta = {'url':url, 'title':title, 'chunk_id': f"{hash(url)}_{i}", 'fetched_at': int(time.time())}
            all_chunks.append({'text': c, 'meta': meta})

    if not all_chunks:
        return {'pagesCrawled': len(pages), 'pagesSkipped':0, 'chunksAdded':0, 'totalChunks':0}

    texts = [c['text'] for c in all_chunks]
    embeddings = embed_texts(texts, model=embedding_model)
    dim = len(embeddings[0])
    store = FaissStore(dim, index_path, docstore_path)
    store.load_or_create()
    docs = [Doc(t, m['meta']) for t, m in zip(texts, all_chunks)]
    store.upsert(embeddings, docs)
    store.persist()

    return {'pagesCrawled': len(pages), 'pagesSkipped':0, 'chunksAdded': len(texts), 'totalChunks': len(store.docstore)}
