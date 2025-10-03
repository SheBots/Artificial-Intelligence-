import fetch from 'node-fetch';

const PROVIDER = (process.env.EMBEDDINGS_PROVIDER || 'gemini').toLowerCase();
const MODEL = process.env.EMBEDDINGS_MODEL || 'text-embedding-004';
const API_KEY = process.env.MODEL_API_KEY || process.env.EMBEDDINGS_API_KEY;

async function embedWithGemini(texts) {
  // Use Google's Generative API REST pattern (simplified)
  if (!API_KEY) throw new Error('EMBEDDINGS API key not configured');

  const url = `https://api.generative.google/v1beta2/models/${MODEL}:embedText`;
  const resp = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${API_KEY}` },
    body: JSON.stringify({ instances: texts.map(t => ({ content: t })) })
  });
  if (!resp.ok) throw new Error(`Embedding error: ${resp.status}`);
  const json = await resp.json();
  // Assume response contains embeddings in json.predictions[i].embedding
  return json.predictions.map(p => p.embedding);
}

async function embedWithOpenAI(texts) {
  if (!API_KEY) throw new Error('EMBEDDINGS API key not configured');
  const url = `https://api.openai.com/v1/embeddings`;
  const resp = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${API_KEY}` },
    body: JSON.stringify({ model: MODEL, input: texts })
  });
  if (!resp.ok) throw new Error(`Embedding error: ${resp.status}`);
  const json = await resp.json();
  return json.data.map(d => d.embedding);
}

export async function embedTexts(texts) {
  if (!Array.isArray(texts)) throw new Error('texts must be array');
  // If no API key is provided, provide a deterministic mock embedding for local dev/testing
  if (!API_KEY) {
    console.warn('EMBEDDINGS API key not configured — using deterministic mock embeddings for dev');
    return texts.map(t => embedMock(t));
  }

  if (PROVIDER === 'openai') return embedWithOpenAI(texts);
  // default to gemini
  return embedWithGemini(texts);
}

function embedMock(text, dim = 256) {
  // Simple deterministic embedding: sum char codes into buckets, then normalize
  const vec = new Array(dim).fill(0);
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i);
    vec[i % dim] += code;
  }
  // normalize
  let norm = 0;
  for (let i = 0; i < dim; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < dim; i++) vec[i] = vec[i] / norm;
  return vec;
}
