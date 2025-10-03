import fetch from 'node-fetch';
import { JSDOM } from 'jsdom';
import { Readability } from '@mozilla/readability';
import { getRobotsFor, canFetch } from './robots.js';
import { setTimeout as delay } from 'timers/promises';

const USE_PLAYWRIGHT = (process.env.USE_PLAYWRIGHT || 'false').toLowerCase() === 'true';

// Minimal Playwright dynamic render path (optional)
async function renderWithPlaywright(url, timeoutMs) {
  // Lazy import to avoid heavy dependency unless enabled
  const { chromium } = await import('playwright');
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  await page.setUserAgent(process.env.CRAWL_USER_AGENT || 'SheBotsCrawler/1.0');
  await page.goto(url, { waitUntil: 'networkidle', timeout: timeoutMs });
  const html = await page.content();
  await browser.close();
  return html;
}

export async function fetchPage(url, opts = {}) {
  const timeoutMs = Number(process.env.CRAWL_TIMEOUT_MS || opts.timeoutMs || 15000);
  const userAgent = process.env.CRAWL_USER_AGENT || 'SheBotsCrawler/1.0';
  const maxSize = Number(process.env.CRAWL_MAX_CONTENT_BYTES || opts.maxContentBytes || 1024 * 1024 * 2); // 2MB

  // Check robots cache and rules (fetch robots if missing)
  const robots = await getRobotsFor(url, { userAgent });
  if (!canFetch(url, { userAgent })) {
    return { ok: false, reason: 'blocked-by-robots' };
  }

  // politeness: optional per-host crawl delay
  const delayMs = Number(process.env.CRAWL_DELAY_MS || opts.delayMs || 1500);
  await delay(delayMs);

  try {
    let html = null;

    if (USE_PLAYWRIGHT) {
      html = await renderWithPlaywright(url, timeoutMs);
    } else {
      const res = await fetch(url, { headers: { 'User-Agent': userAgent }, timeout: timeoutMs });
      const contentType = res.headers.get('content-type') || '';
      if (!res.ok) return { ok: false, reason: `http-${res.status}` };
      if (!contentType.includes('text/html')) return { ok: false, reason: 'not-html' };

      // stream size guard: read as text but guard length
      const text = await res.text();
      if (text.length > maxSize) return { ok: false, reason: 'max-content-size' };
      html = text;
    }

    return { ok: true, url, html };
  } catch (err) {
    return { ok: false, reason: 'fetch-error', error: err.message };
  }
}
