import os, time, json, logging, requests
from rag.loader import crawl, extract_pdf, extract_docx, extract_hwp, extract_image_ocr
from rag.clean import clean_text
from rag.splitter import split_text
from rag.embeddings import embed_texts
from rag.store import FaissStore, Doc

# NEW: manual ingestion lists
from rag.data import MANUAL_URLS, PDF_FILES, DOCX_FILES, TEXT_FILES

logger = logging.getLogger(__name__)


# -----------------------------
# ATTACHMENT PROCESSING
# -----------------------------
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

    # Clean + split into chunks
    cleaned = clean_text(text)
    chunks = split_text(cleaned)

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



# ---------------------------------------------------------
# MAIN INGEST FUNCTION
# ---------------------------------------------------------
def ingest(start_urls, allowlist, index_path, docstore_path, embedding_model,
           max_pages=300, max_depth=2, delay_ms=1500):

    # Enable auto crawling (disabled for manual testing)
    #pages = crawl(start_urls, allowlist, max_pages=max_pages, max_depth=max_depth, delay_ms=delay_ms)
    
    pages = []
    all_chunks = []
    attachment_count = 0
    html_chunk_count = 0

    # -----------------------------
    # 1) PROCESS CRAWLED HTML PAGES
    # -----------------------------
    for p in pages:
        url = p['url']
        title = p.get('title', '')
        text = clean_text(p['text'])
        chunks = split_text(text)

        # HTML chunks
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

        # Attachments inside crawled pages
        attachments = p.get('attachments', [])
        for att in attachments:
            try:
                att_chunks = process_attachment(att, url, title)
                all_chunks.extend(att_chunks)
                attachment_count += 1
            except Exception as e:
                logger.error(f"Failed to process attachment {att.get('path')}: {e}")


    # ======================================================
    # 2) MANUAL INGESTION SECTION
    # ======================================================

    # 2A) Manual URLs
    for url in MANUAL_URLS:
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                html = clean_text(r.text)
                chunks = split_text(html)
                for i, c in enumerate(chunks):
                    meta = {
                        'url': url,
                        'title': f"manual:{url}",
                        'chunk_id': f"{hash((url, 'manual_url'))}_{i}",
                        'source_type': 'manual_url',
                        'fetched_at': int(time.time())
                    }
                    all_chunks.append({'text': c, 'meta': meta})
            else:
                logger.error(f"Manual URL failed {url} (status {r.status_code})")
        except Exception as e:
            logger.error(f"Manual URL fetch error {url}: {e}")


    # 2B) Manual PDFs
    for pdf_path in PDF_FILES:
        try:
            text = extract_pdf(pdf_path)
            cleaned = clean_text(text)
            chunks = split_text(cleaned)
            for i, c in enumerate(chunks):
                meta = {
                    'url': pdf_path,
                    'title': "manual_pdf",
                    'chunk_id': f"{hash((pdf_path, 'manual_pdf'))}_{i}",
                    'source_type': 'manual_pdf',
                    'fetched_at': int(time.time())
                }
                all_chunks.append({'text': c, 'meta': meta})
        except Exception as e:
            logger.error(f"Failed to ingest PDF {pdf_path}: {e}")


    # 2C) Manual DOCX files
    for docx_path in DOCX_FILES:
        try:
            text = extract_docx(docx_path)
            cleaned = clean_text(text)
            chunks = split_text(cleaned)
            for i, c in enumerate(chunks):
                meta = {
                    'url': docx_path,
                    'title': "manual_docx",
                    'chunk_id': f"{hash((docx_path, 'manual_docx'))}_{i}",
                    'source_type': 'manual_docx',
                    'fetched_at': int(time.time())
                }
                all_chunks.append({'text': c, 'meta': meta})
        except Exception as e:
            logger.error(f"Failed to ingest DOCX {docx_path}: {e}")


    # 2D) Manual TEXT files
    for txt_path in TEXT_FILES:
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read()
            cleaned = clean_text(text)
            chunks = split_text(cleaned)
            for i, c in enumerate(chunks):
                meta = {
                    'url': txt_path,
                    'title': "manual_text",
                    'chunk_id': f"{hash((txt_path, 'manual_text'))}_{i}",
                    'source_type': 'manual_text',
                    'fetched_at': int(time.time())
                }
                all_chunks.append({'text': c, 'meta': meta})
        except Exception as e:
            logger.error(f"Failed to ingest TEXT file {txt_path}: {e}")


    # -----------------------------
    # SAFETY CHECK
    # -----------------------------
    if not all_chunks:
        return {
            'pagesCrawled': len(pages),
            'chunksAdded': 0,
            'totalChunks': 0,
            'attachmentsProcessed': attachment_count,
            'htmlChunks': 0,
            'manualChunks': 0
        }


    # -----------------------------
    # 3) EMBED + SAVE TO FAISS
    # -----------------------------
    texts = [c['text'] for c in all_chunks]
    embeddings = embed_texts(texts, model=embedding_model)
    dim = len(embeddings[0])

    store = FaissStore(dim, index_path, docstore_path)
    store.load_or_create()
    docs = [Doc(t, c['meta']) for t, c in zip(texts, all_chunks)]
    store.upsert(embeddings, docs)
    store.persist()

    return {
        'pagesCrawled': len(pages),
        'chunksAdded': len(all_chunks),
        'totalChunks': len(store.docstore),
        'attachmentsProcessed': attachment_count,
        'htmlChunks': html_chunk_count,
        'manualChunks': len(all_chunks) - html_chunk_count
    }
