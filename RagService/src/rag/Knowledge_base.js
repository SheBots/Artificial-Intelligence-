import fetch from 'node-fetch';
import { JSDOM } from 'jsdom';
import { Readability } from '@mozilla/readability';
import { TextSplitter } from 'langchain/text_splitter';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import Chroma from 'chromadb';
import dotenv from 'dotenv';

dotenv.config();

const DEFAULT_CHUNK_SIZE = parseInt(process.env.RAG_CHUNK_SIZE || '1000', 10);
const DEFAULT_CHUNK_OVERLAP = parseInt(process.env.RAG_CHUNK_OVERLAP || '200', 10);

function extractTextFromHtml(html, url) {
  try {
    const dom = new JSDOM(html, { url });
    const reader = new Readability(dom.window.document);
    const article = reader.parse();
    return (article && article.textContent) || dom.window.document.body.textContent || '';
  } catch (err) {
    return '';
  }
}

async function fetchPage(url, timeout = 15000) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);
  try {
    const res = await fetch(url, { signal: controller.signal, headers: { 'User-Agent': process.env.CRAWL_USER_AGENT || 'SheBotsRAG/1.0' } });
    clearTimeout(id);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.text();
  } catch (err) {
    clearTimeout(id);
    throw err;
  }
}

export async function loadAndIndexUrls(urls = [], options = {}) {
  if (!Array.isArray(urls) || urls.length === 0) throw new Error('urls must be a non-empty array');

  // Initialize Chroma client
  const chromaClient = new Chroma.ChromaClient();
  const collectionName = options.collectionName || process.env.CHROMA_COLLECTION || 'shebots_rag';

  const collection = await chromaClient.createCollection({ name: collectionName }).catch(async () => {
    return await chromaClient.getCollection({ name: collectionName });
  });

  const splitter = new TextSplitter({ chunkSize: DEFAULT_CHUNK_SIZE, chunkOverlap: DEFAULT_CHUNK_OVERLAP });

  // Determine embedding provider
  const provider = (process.env.RAG_EMBEDDING_PROVIDER || 'openai').toLowerCase();
  let embedder;
  if (provider === 'openai') {
    embedder = new OpenAIEmbeddings({ openAIApiKey: process.env.OPENAI_API_KEY });
  } else {
    // fallback to OpenAI embeddings for now
    embedder = new OpenAIEmbeddings({ openAIApiKey: process.env.OPENAI_API_KEY });
  }

  let totalChunks = 0;

  for (const url of urls) {
    try {
      const html = await fetchPage(url, parseInt(process.env.CRAWL_TIMEOUT_MS || '30000', 10));
      const text = extractTextFromHtml(html, url);
      if (!text || text.trim().length < 50) continue;

      // split
      const docs = splitter.splitText(text);

      // embed and upsert in batches
      for (let i = 0; i < docs.length; i += 16) {
        const batch = docs.slice(i, i + 16);
        const texts = batch.map((t) => t);
        const metadatas = batch.map((chunk, idx) => ({
          source: url,
          index: i + idx,
          length: chunk.length,
        }));

        const embeddings = await embedder.embedDocuments(texts);

        const ids = embeddings.map((_, idx) => `${Date.now()}-${Math.random().toString(36).slice(2, 8)}-${i + idx}`);

        await collection.upsert({
          ids,
          embeddings,
          metadatas,
          documents: texts,
        });

        totalChunks += texts.length;
      }
    } catch (err) {
      console.warn(`Failed to index ${url}: ${err.message}`);
      continue;
    }
  }

  return { success: true, collection: collectionName, totalChunks };
}

export async function queryCollection(query, k = 5) {
  const chromaClient = new Chroma.ChromaClient();
  const collectionName = process.env.CHROMA_COLLECTION || 'shebots_rag';
  const collection = await chromaClient.getCollection({ name: collectionName });
  if (!collection) return { results: [] };

  // embed query
  const embedder = new OpenAIEmbeddings({ openAIApiKey: process.env.OPENAI_API_KEY });
  const qVec = await embedder.embedQuery(query);

  const resp = await collection.query({ query_embeddings: [qVec], n_results: k, include: ['metadatas', 'documents', 'distances'] });
  // normalize results
  const results = (resp?.results?.[0] || []).map((r) => ({
    id: r.id,
    score: 1 - r.distance,
    metadata: r.metadata,
    text: r.document,
  }));

  return { results };
}
