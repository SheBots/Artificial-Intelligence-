import { fetchPage } from '../scraper/fetchPage.js';
import { extractFromHtml } from '../scraper/extract.js';
import { chunkText } from './splitter.js';
import { embedTexts } from './embedder.js';
import { upsertBulk, computeHash } from './store.js';
import { URL } from 'url';

function normalizeUrl(base, link) {
  try {
    return new URL(link, base).toString().split('#')[0];
  } catch (e) { return null; }
}

function hostAllowed(url, allowlist) {
  try {
    const u = new URL(url);
    return allowlist.some(a => new URL(a).origin === u.origin);
  } catch (e) { return false; }
}

export async function crawl({ seeds = [], allowlist = [], maxPages = 200, maxDepth = 2 }) {
  const visited = new Set();
  const queue = [];
  for (const s of seeds) queue.push({ url: s, depth: 0 });

  let pagesCrawled = 0;
  let pagesSkipped = 0;
  let chunksAdded = 0;

  while (queue.length > 0 && pagesCrawled < maxPages) {
    const { url, depth } = queue.shift();
    if (visited.has(url)) continue;
    visited.add(url);

    if (!hostAllowed(url, allowlist)) {
      pagesSkipped++;
      continue;
    }

    // fetch
    const res = await fetchPage(url);
    if (!res.ok) {
      pagesSkipped++;
      continue;
    }

    const extracted = extractFromHtml(res.html, url);
    if (!extracted) {
      pagesSkipped++;
      continue;
    }

    // chunk
    const chunks = chunkText(extracted.text, { target: 700, overlap: 120 });
    const texts = chunks.map((c, i) => `${extracted.title}\n\n${c}`);

    // embed
    let vectors = [];
    try {
      vectors = await embedTexts(texts);
    } catch (e) {
      pagesSkipped++;
      continue;
    }

    // prepare entries
    const entries = texts.map((t, i) => ({
      id: `${computeHash(url)}::${i}`,
      url,
      title: extracted.title || '',
      text: t,
      vector: vectors[i],
      hash: computeHash(t),
      fetchedAt: Date.now()
    }));

    const result = upsertBulk(entries);
    chunksAdded += result.added;
    pagesCrawled++;

    // enqueue links if depth+1 <= maxDepth
    if (depth + 1 <= maxDepth) {
      // extract links
      const linkRegex = /href=["']([^"'#]+)["']/g;
      const matches = Array.from(res.html.matchAll(linkRegex)).map(m => m[1]);
      for (const l of matches) {
        const abs = normalizeUrl(url, l);
        if (!abs) continue;
        if (visited.has(abs)) continue;
        if (!hostAllowed(abs, allowlist)) continue;
        queue.push({ url: abs, depth: depth + 1 });
      }
    }
  }

  const totalChunks = (function(){ const all = require('fs').existsSync(process.env.RAG_STORE || './data/web_index.jsonl') ? require('fs').readFileSync(process.env.RAG_STORE || './data/web_index.jsonl','utf8').split(/\r?\n/).filter(Boolean).length : 0; return all; })();

  return { pagesCrawled, pagesSkipped, chunksAdded, totalChunks };
}
