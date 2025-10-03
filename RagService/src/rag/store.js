import fs from 'fs';
import { createHash } from 'crypto';

const STORE_PATH = process.env.RAG_STORE || './data/web_index.jsonl';
const MIN_SCORE = Number(process.env.RAG_MIN_SCORE || 0.65);

// Ensure directory exists
try { fs.mkdirSync(require('path').dirname(STORE_PATH), { recursive: true }); } catch (e) {}

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

function readAll() {
  if (!fs.existsSync(STORE_PATH)) return [];
  const lines = fs.readFileSync(STORE_PATH, 'utf8').split(/\r?\n/).filter(Boolean);
  return lines.map(l => JSON.parse(l));
}

function writeAll(rows) {
  const tmp = STORE_PATH + '.tmp';
  fs.writeFileSync(tmp, rows.map(r => JSON.stringify(r)).join('\n'));
  fs.renameSync(tmp, STORE_PATH);
}

export function upsertBulk(entries) {
  const existing = readAll();
  const map = new Map(existing.map(e => [`${e.url}::${e.hash}`, e]));
  let added = 0;
  for (const e of entries) {
    const key = `${e.url}::${e.hash}`;
    if (!map.has(key)) {
      map.set(key, e);
      added++;
    }
  }
  const all = Array.from(map.values());
  writeAll(all);
  return { total: all.length, added };
}

export function searchByVector(queryVec, k = 5, minScore = MIN_SCORE) {
  const all = readAll();
  const scored = all.map(r => ({ ...r, score: cosine(queryVec, r.vector) }));
  scored.sort((a,b) => b.score - a.score);
  return scored.filter(s => s.score >= minScore).slice(0, k).map(s => ({ url: s.url, title: s.title, text: s.text, score: s.score }));
}

export function computeHash(text) {
  return createHash('sha256').update(text).digest('hex');
}
