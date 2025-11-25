import os, time, json, logging
from rag.loader import crawl, extract_pdf, extract_docx, extract_hwp, extract_image_ocr
from rag.clean import clean_text
from rag.splitter import split_text
from rag.embeddings import embed_texts
from rag.store import FaissStore, Doc

logger = logging.getLogger(__name__)

def process_attachment(attachment, page_url, page_title):
    """Extract text from attachment based on type."""
    att_type = attachment['type']
    filepath = attachment['path']
    
    logger.info(f"Processing {att_type} attachment: {filepath}")
    
    text = ""
    if att_type == 'pdf':
        text = extract_pdf(filepath)
    elif att_type == 'docx':
        text = extract_docx(filepath)
    elif att_type == 'hwp':
        text = extract_hwp(filepath)
    elif att_type == 'image':
        text = extract_image_ocr(filepath)
    
    if not text or len(text.strip()) < 50:
        logger.warning(f"Insufficient text extracted from {filepath}")
        return []
    
    # Clean and split
    cleaned = clean_text(text)
    chunks = split_text(cleaned)
    
    # Create chunks with metadata
    result_chunks = []
    for i, chunk in enumerate(chunks):
        meta = {
            'url': page_url,
            'title': page_title,
            'chunk_id': f"{hash(filepath)}_{i}",
            'source_type': att_type,
            'attachment_url': attachment['url'],
            'attachment_path': filepath,
            'fetched_at': int(time.time())
        }
        result_chunks.append({'text': chunk, 'meta': meta})
    
    logger.info(f"Extracted {len(result_chunks)} chunks from {att_type} attachment")
    return result_chunks


def ingest(start_urls, allowlist, index_path, docstore_path, embedding_model, max_pages=300, max_depth=2, delay_ms=1500):
    pages = crawl(start_urls, allowlist, max_pages=max_pages, max_depth=max_depth, delay_ms=delay_ms)
    store = None
    all_chunks = []
    
    attachment_count = 0
    html_chunk_count = 0
    
    for p in pages:
        url = p['url']
        title = p.get('title','')
        text = clean_text(p['text'])
        chunks = split_text(text)
        
        # Process HTML chunks
        for i, c in enumerate(chunks):
            meta = {
                'url': url,
                'title': title,
                'chunk_id': f"{hash(url)}_{i}",
                'source_type': 'html',
                'fetched_at': int(time.time())
            }
            all_chunks.append({'text': c, 'meta': meta})
        html_chunk_count += len(chunks)
        
        # Process attachments
        attachments = p.get('attachments', [])
        for att in attachments:
            try:
                att_chunks = process_attachment(att, url, title)
                all_chunks.extend(att_chunks)
                attachment_count += 1
            except Exception as e:
                logger.error(f"Failed to process attachment {att.get('path')}: {e}")

    if not all_chunks:
        return {
            'pagesCrawled': len(pages),
            'pagesSkipped': 0,
            'chunksAdded': 0,
            'totalChunks': 0,
            'attachmentsProcessed': 0,
            'htmlChunks': 0
        }

    texts = [c['text'] for c in all_chunks]
    embeddings = embed_texts(texts, model=embedding_model)
    dim = len(embeddings[0])
    store = FaissStore(dim, index_path, docstore_path)
    store.load_or_create()
    docs = [Doc(t, m['meta']) for t, m in zip(texts, all_chunks)]
    store.upsert(embeddings, docs)
    store.persist()

    return {
        'pagesCrawled': len(pages),
        'pagesSkipped': 0,
        'chunksAdded': len(texts),
        'totalChunks': len(store.docstore),
        'attachmentsProcessed': attachment_count,
        'htmlChunks': html_chunk_count,
        'attachmentChunks': len(texts) - html_chunk_count
    }
