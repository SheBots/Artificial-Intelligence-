import express from 'express';
import { crawl } from '../rag/crawl.js';

const router = express.Router();

router.post('/crawl', async (req, res) => {
  try {
    const envAllowlist = (process.env.CRAWL_ALLOWLIST || '').split(',').map(s => s.trim()).filter(Boolean);
    const { seeds = [], maxPages, maxDepth } = req.body || {};

    // validate seeds: must be subset of allowlist origins
    const allowlist = envAllowlist;
    const seedsToUse = (seeds.length ? seeds : envAllowlist.slice(0,5)).map(s => s.trim());

    for (const s of seedsToUse) {
      if (!allowlist.some(a => new URL(a).origin === new URL(s).origin)) {
        return res.status(400).json({ error: 'Seeds must be within configured allowlist' });
      }
    }

    const summary = await crawl({ seeds: seedsToUse, allowlist, maxPages: maxPages || Number(process.env.CRAWL_MAX_PAGES || 200), maxDepth: maxDepth || Number(process.env.CRAWL_MAX_DEPTH || 2) });
    res.json({ ok: true, summary });
  } catch (e) {
    console.error('Crawl error', e);
    res.status(500).json({ error: e.message });
  }
});

export default router;
