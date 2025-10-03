import { JSDOM } from 'jsdom';
import { load } from 'cheerio';
import { Readability } from '@mozilla/readability';
import { createHash } from 'crypto';

function heuristicClean(html) {
  const $ = load(html);
  // Remove scripts, styles, nav, header, footer, ads
  $('script,noscript,style,header,footer,nav,iframe,aside').remove();
  // Remove common ad selectors
  $('[class*="ad"]').remove();
  $('[id*="ad"]').remove();

  // Extract title
  const title = ($('meta[property="og:title"]').attr('content') || $('title').text() || '').trim();

  // Attempt to get main content heuristically: largest <p> block container
  let text = '';
  const candidates = [];
  $('body *').each((i, el) => {
    const $el = $(el);
    const txt = $el.text().trim();
    if (txt.length > 200) candidates.push({ el, len: txt.length, text: txt });
  });
  candidates.sort((a, b) => b.len - a.len);
  if (candidates.length > 0) {
    text = candidates.slice(0, 3).map(c => c.text).join('\n\n');
  } else {
    text = $('body').text().replace(/\s+/g, ' ').trim();
  }

  return { title, text };
}

export function extractFromHtml(html, url) {
  // Use JSDOM + Readability primarily
  try {
    const dom = new JSDOM(html, { url });
    const reader = new Readability(dom.window.document);
    const article = reader.parse();
    if (article && article.textContent && article.textContent.length > 500) {
      const title = article.title || (dom.window.document.querySelector('title')?.textContent || '').trim();
      const text = article.textContent.trim();
      const hash = createHash('sha256').update(text).digest('hex');
      return { title, text, hash };
    }
  } catch (e) {
    // Fallthrough to heuristic
  }

  // Heuristic fallback
  const { title, text } = heuristicClean(html);
  if (!text || text.length < 500) return null;
  const hash = createHash('sha256').update(text).digest('hex');
  return { title: title || '', text, hash };
}
