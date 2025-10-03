import express from 'express';
import dotenv from 'dotenv';
import cors from 'cors';
import rateLimit from 'express-rate-limit';
import crawlRouter from './src/routes/crawl.js';
import searchRouter from './src/routes/search.js';

dotenv.config();

const app = express();
const PORT = process.env.RAG_PORT || 3001;
const ALLOWED_ORIGIN = process.env.ALLOWED_ORIGIN || 'http://localhost:5173';

app.use(express.json({ limit: '1mb' }));
app.use(cors({ origin: ALLOWED_ORIGIN, credentials: true }));

const limiter = rateLimit({ windowMs: 60*1000, max: 60 });
app.use('/api', limiter);

// Simple API key middleware for service-to-service auth (optional)
app.use((req, res, next) => {
  const key = process.env.RAG_API_KEY;
  if (!key) return next();
  const auth = req.headers['authorization'];
  if (!auth || String(auth).indexOf('Bearer ') !== 0) return res.status(401).json({ error: 'Unauthorized' });
  const token = auth.split(' ')[1];
  if (token !== key) return res.status(401).json({ error: 'Unauthorized' });
  next();
});

app.use('/api', crawlRouter);
app.use('/api', searchRouter);

app.get('/api/health', (req, res) => res.json({ ok: true, rag: true }));

app.listen(PORT, () => {
  console.log(`RAG service listening on http://localhost:${PORT}`);
});
