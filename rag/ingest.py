# ingest.py — ingestion pipeline for SheBots RAG
# - uses strict HTML cleaner for KNU CSE pages
# - supports manual URLs + PDF/DOCX/TXT files
# - stores clean chunks into FAISS + docstore

import os
import time
import json
import logging
import requests

from .loader import (
    crawl,
    extract_pdf,
    extract_docx,
    extract_hwp,
    extract_image_ocr,
)
from .clean import clean_html_strict, clean_text
from .splitter import split_text
from .embeddings import embed_texts
from .store import FaissStore, Doc

# Manual ingestion lists
from .data import MANUAL_URLS, PDF_FILES, DOCX_FILES, TEXT_FILES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# ATTACHMENT PROCESSING
# ---------------------------------------------------------
def process_attachment(attachment, page_url, page_title):
    """Extract text from an attached file and convert into clean chunks."""
    att_type = attachment["type"]
    filepath = attachment["path"]

    logger.info(f"Processing {att_type} attachment: {filepath}")

    text = ""
    if att_type == "pdf":
        text = extract_pdf(filepath)
    elif att_type == "docx":
        text = extract_docx(filepath)
    elif att_type == "hwp":
        text = extract_hwp(filepath)
    elif att_type == "image":
        text = extract_image_ocr(filepath)

    # Skip tiny or empty attachments
    if not text or len(text.strip()) < 50:
        logger.warning(f"Insufficient text extracted from {filepath}")
        return []

    # Plain-text cleaning (attachments are usually not full HTML pages)
    cleaned = clean_text(text)
    chunks = split_text(cleaned)

    result_chunks = []
    for i, chunk in enumerate(chunks):
        meta = {
            "url": page_url,
            "title": page_title,
            "chunk_id": f"{hash(filepath)}_{i}",
            "source_type": att_type,
            "attachment_url": attachment.get("url"),
            "attachment_path": filepath,
            "fetched_at": int(time.time()),
        }
        result_chunks.append({"text": chunk, "meta": meta})

    logger.info(f"Extracted {len(result_chunks)} chunks from attachment {filepath}")
    return result_chunks


# ---------------------------------------------------------
# MAIN INGEST FUNCTION
# ---------------------------------------------------------
def ingest(
    start_urls,
    allowlist,
    index_path,
    docstore_path,
    embedding_model,
    max_pages=300,
    max_depth=2,
    delay_ms=1500,
):
    """
    Ingests:
      - (optionally) crawled HTML pages (currently disabled for safety)
      - manually listed URLs (MANUAL_URLS)
      - PDFs / DOCX / TXT listed in data.py

    Produces:
      - FAISS index at index_path
      - docstore.jsonl at docstore_path
    """

    # If you want auto-crawling later, re-enable this:
    # pages = crawl(start_urls, allowlist,
    #               max_pages=max_pages, max_depth=max_depth,
    #               delay_ms=delay_ms)

    pages = []  # manual mode for now

    all_chunks = []
    attachment_count = 0
    html_chunk_count = 0

    # ---------------------------------------------------------
    # 1) PROCESS CRAWLED HTML PAGES (currently none)
    # ---------------------------------------------------------
    for p in pages:
        url = p["url"]
        title = p.get("title", "")

        # Use strict cleaner for KNU site HTML
        text = clean_html_strict(p["text"])
        chunks = split_text(text)

        # HTML chunks
        for i, c in enumerate(chunks):
            meta = {
                "url": url,
                "title": title,
                "chunk_id": f"{hash(url)}_{i}",
                "source_type": "html",
                "fetched_at": int(time.time()),
            }
            all_chunks.append({"text": c, "meta": meta})
        html_chunk_count += len(chunks)

        # Attachments inside crawled pages
        attachments = p.get("attachments", [])
        for att in attachments:
            try:
                att_chunks = process_attachment(att, url, title)
                all_chunks.extend(att_chunks)
                attachment_count += 1
            except Exception as e:
                logger.error(
                    f"Failed to process attachment {att.get('path')}: {e}"
                )

    # =========================================================
    # 2) MANUAL INGESTION SECTION
    # =========================================================

    # 2A) Manual URLs (main source of FAQ / notice content)
    logger.info("Starting manual URL ingestion...")
    for url in MANUAL_URLS:
        try:
            r = requests.get(url, timeout=12)
            if r.status_code != 200:
                logger.error(f"Manual URL failed {url} (status {r.status_code})")
                continue

            html_clean = clean_html_strict(r.text)
            chunks = split_text(html_clean)

            for i, c in enumerate(chunks):
                meta = {
                    "url": url,
                    "title": f"manual:{url}",
                    "chunk_id": f"{hash((url, 'manual_url'))}_{i}",
                    "source_type": "manual_url",
                    "fetched_at": int(time.time()),
                }
                all_chunks.append({"text": c, "meta": meta})

        except Exception as e:
            logger.error(f"Manual URL fetch error {url}: {e}")

    # 2B) Manual PDFs
    logger.info("Ingesting manual PDFs...")
    for pdf_path in PDF_FILES:
        try:
            text = extract_pdf(pdf_path)
            cleaned = clean_text(text)
            chunks = split_text(cleaned)

            for i, c in enumerate(chunks):
                meta = {
                    "url": pdf_path,
                    "title": "manual_pdf",
                    "chunk_id": f"{hash((pdf_path, 'manual_pdf'))}_{i}",
                    "source_type": "manual_pdf",
                    "fetched_at": int(time.time()),
                }
                all_chunks.append({"text": c, "meta": meta})

        except Exception as e:
            logger.error(f"Failed to ingest PDF {pdf_path}: {e}")

    # 2C) Manual DOCX files
    logger.info("Ingesting manual DOCX files...")
    for docx_path in DOCX_FILES:
        try:
            text = extract_docx(docx_path)
            cleaned = clean_text(text)
            chunks = split_text(cleaned)

            for i, c in enumerate(chunks):
                meta = {
                    "url": docx_path,
                    "title": "manual_docx",
                    "chunk_id": f"{hash((docx_path, 'manual_docx'))}_{i}",
                    "source_type": "manual_docx",
                    "fetched_at": int(time.time()),
                }
                all_chunks.append({"text": c, "meta": meta})

        except Exception as e:
            logger.error(f"Failed to ingest DOCX {docx_path}: {e}")

    # 2D) Manual TEXT files
    logger.info("Ingesting manual TEXT files...")
    for txt_path in TEXT_FILES:
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read()
            cleaned = clean_text(text)
            chunks = split_text(cleaned)

            for i, c in enumerate(chunks):
                meta = {
                    "url": txt_path,
                    "title": "manual_text",
                    "chunk_id": f"{hash((txt_path, 'manual_text'))}_{i}",
                    "source_type": "manual_text",
                    "fetched_at": int(time.time()),
                }
                all_chunks.append({"text": c, "meta": meta})

        except Exception as e:
            logger.error(f"Failed to ingest TEXT file {txt_path}: {e}")

    # ---------------------------------------------------------
    # SAFETY CHECK
    # ---------------------------------------------------------
    if not all_chunks:
        logger.warning("No chunks generated during ingest.")
        return {
            "pagesCrawled": len(pages),
            "chunksAdded": 0,
            "totalChunks": 0,
            "attachmentsProcessed": attachment_count,
            "htmlChunks": 0,
            "manualChunks": 0,
        }

    # ---------------------------------------------------------
    # 3) EMBED + SAVE TO FAISS
    # ---------------------------------------------------------
    logger.info(f"Embedding {len(all_chunks)} chunks...")
    texts = [c["text"] for c in all_chunks]
    embeddings = embed_texts(texts, model=embedding_model)
    dim = len(embeddings[0])

    store = FaissStore(dim, index_path, docstore_path)
    store.load_or_create()

    docs = [Doc(t, c["meta"]) for t, c in zip(texts, all_chunks)]
    store.upsert(embeddings, docs)
    store.persist()

    logger.info("Ingestion complete.")

    return {
        "pagesCrawled": len(pages),
        "chunksAdded": len(all_chunks),
        "totalChunks": len(store.docstore),
        "attachmentsProcessed": attachment_count,
        "htmlChunks": html_chunk_count,
        "manualChunks": len(all_chunks) - html_chunk_count,
    }
