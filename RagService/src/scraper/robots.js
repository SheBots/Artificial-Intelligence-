import fetch from 'node-fetch';
import { URL } from 'url';

// Simple robots.txt fetcher + parser cache
const cache = new Map(); // host -> { fetchedAt, rules }

function parseRobotsTxt(text, ua = '*') {
  const lines = text.split(/\r?\n/).map(l => l.trim());
  const rules = { allow: [], disallow: [], crawlDelay: null };
  let currentUserAgents = [];

  for (let line of lines) {
    if (!line || line.startsWith('#')) continue;
    const [rawKey, rawVal] = line.split(':', 2);
    if (!rawKey || !rawVal) continue;
    const key = rawKey.trim().toLowerCase();
    const val = rawVal.trim();

    if (key === 'user-agent') {
      currentUserAgents = [val.toLowerCase()];
    } else if (key === 'disallow') {
      for (const ua of currentUserAgents) {
        if (ua === '*' || ua === '') rules.disallow.push(val);
      }
    } else if (key === 'allow') {
      for (const ua of currentUserAgents) {
        if (ua === '*' || ua === '') rules.allow.push(val);
      }
    } else if (key === 'crawl-delay') {
      const n = Number(val);
      if (!Number.isNaN(n)) rules.crawlDelay = n * 1000;
    }
  }

  return rules;
}

export async function getRobotsFor(url, opts = {}) {
  const { userAgent = 'SheBotsCrawler/1.0' } = opts;
  const u = new URL(url);
  const host = u.origin;

  if (cache.has(host)) return cache.get(host);

  try {
    const robotsUrl = `${host}/robots.txt`;
    const resp = await fetch(robotsUrl, { headers: { 'User-Agent': userAgent }, timeout: 5000 });
    if (!resp.ok) {
      const result = { rules: null, fetchedAt: Date.now() };
      cache.set(host, result);
      return result;
    }

    const txt = await resp.text();
    const rules = parseRobotsTxt(txt);
    const result = { rules, fetchedAt: Date.now() };
    cache.set(host, result);
    return result;
  } catch (err) {
    const result = { rules: null, fetchedAt: Date.now() };
    cache.set(host, result);
    return result;
  }
}

export function canFetch(url, opts = {}) {
  const { userAgent = 'SheBotsCrawler/1.0' } = opts;
  const u = new URL(url);
  const host = u.origin;

  if (!cache.has(host)) return true; // unknown -> allow by default (will fetch robots soon)

  const entry = cache.get(host);
  if (!entry || !entry.rules) return true;

  const path = u.pathname + (u.search || '');
  // Allow rules take precedence over disallow in common implementations; we'll check allow first
  for (const allow of entry.rules.allow) {
    if (allow === '') continue;
    if (path.startsWith(allow)) return true;
  }
  for (const dis of entry.rules.disallow) {
    if (dis === '') continue; // empty disallow means allow all
    if (path.startsWith(dis)) return false;
  }

  return true;
}
