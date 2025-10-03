import express from 'express';
import { embedTexts } from '../rag/embedder.js';
import { searchByVector } from '../rag/store.js';

const router = express.Router();

router.get('/search', async (req, res) => {
  try {
    const q = (req.query.q || '').trim();
    const k = Number(req.query.k || process.env.RAG_TOP_K || 5);
    if (!q) return res.status(400).json({ error: 'q is required' });

    const [vec] = await embedTexts([q]);
    const results = searchByVector(vec, k);
    res.json({ query: q, results });
  } catch (e) {
    console.error('Search error', e);
    res.status(500).json({ error: e.message });
  }
});

export default router;
