import os, time, logging, urllib.parse
from bs4 import BeautifulSoup
import httpx
from urllib.parse import urljoin, urlparse
from collections import deque

logger = logging.getLogger(__name__)

USER_AGENT = "SheBotsRAG/1.0 (+contact@example.com)"

def allowed_by_robots(base_url, path):
    # basic robots check: fetch /robots.txt and look for Disallow lines (simple)
    try:
        robots_url = urljoin(base_url, '/robots.txt')
        r = httpx.get(robots_url, timeout=10.0)
        if r.status_code != 200:
            return True
        text = r.text
        # naive check
        for line in text.splitlines():
            line = line.strip()
            if line.lower().startswith('user-agent:'):
                continue
            if line.lower().startswith('disallow:'):
                path_dis = line.split(':',1)[1].strip()
                if path.startswith(path_dis):
                    return False
        return True
    except Exception:
        return True


def fetch_page(url, timeout=15.0):
    headers = {'User-Agent': USER_AGENT}
    try:
        r = httpx.get(url, headers=headers, timeout=timeout)
        if r.status_code != 200:
            return None
        ctype = r.headers.get('content-type','')
        if 'text/html' not in ctype:
            return None
        return r.text
    except Exception as e:
        logger.info('fetch error %s %s', url, e)
        return None


def extract_text_and_title(html, base_url, url):
    soup = BeautifulSoup(html, 'lxml')
    title = (soup.title.string.strip() if soup.title and soup.title.string else '')
    # naive main text extraction: join paragraphs
    paragraphs = soup.find_all('p')
    text = '\n\n'.join([p.get_text(separator=' ', strip=True) for p in paragraphs])
    if len(text) < 400:
        # try fallback: get text from main
        main = soup.find('main')
        if main:
            text = main.get_text(separator=' ', strip=True)
    return title, text


def crawl(start_urls, allowlist, max_pages=300, max_depth=2, delay_ms=1500):
    seen = set()
    q = deque()
    for u in start_urls:
        q.append((u,0))

    results = []
    while q and len(results) < max_pages:
        url, depth = q.popleft()
        if url in seen:
            continue
        seen.add(url)
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if not any(url.startswith(a) for a in allowlist):
            continue
        if not allowed_by_robots(base, parsed.path):
            logger.info('robots disallow %s', url)
            continue

        html = fetch_page(url)
        time.sleep(delay_ms/1000)
        if not html:
            continue
        title, text = extract_text_and_title(html, base, url)
        if not text or len(text) < 400:
            continue
        results.append({'url':url, 'title':title, 'text':text})

        if depth < max_depth:
            soup = BeautifulSoup(html, 'lxml')
            for a in soup.find_all('a', href=True):
                href = a['href']
                if href.startswith('mailto:') or href.startswith('tel:'):
                    continue
                new = urljoin(url, href)
                # normalize
                p = urlparse(new)
                new_norm = p._replace(fragment='').geturl()
                if new_norm not in seen and any(new_norm.startswith(a) for a in allowlist):
                    q.append((new_norm, depth+1))

    return results
