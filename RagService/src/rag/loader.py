import os, time, logging, urllib.parse
from bs4 import BeautifulSoup
import httpx
from urllib.parse import urljoin, urlparse
from collections import deque
from pathlib import Path
import mimetypes

logger = logging.getLogger(__name__)

USER_AGENT = "SheBotsRAG/1.0 (+contact@example.com)"

# ========================= FILE EXTRACTION HELPERS =========================

def extract_pdf(path):
    """Extract text from PDF file."""
    try:
        import PyPDF2
        text = []
        with open(path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                content = page.extract_text()
                if content:
                    text.append(content)
        return '\n\n'.join(text)
    except Exception as e:
        logger.error(f"PDF extraction error {path}: {e}")
        return ""


def extract_docx(path):
    """Extract text from DOCX file."""
    try:
        from docx import Document
        doc = Document(path)
        text = []
        for para in doc.paragraphs:
            if para.text.strip():
                text.append(para.text)
        return '\n\n'.join(text)
    except Exception as e:
        logger.error(f"DOCX extraction error {path}: {e}")
        return ""


def extract_hwp(path):
    """Extract text from HWP file using hwp5txt or fallback."""
    try:
        import subprocess
        result = subprocess.run(['hwp5txt', path], capture_output=True, text=True, timeout=30, encoding='utf-8', errors='replace')
        if result.returncode == 0 and result.stdout:
            return result.stdout
    except Exception as e:
        logger.warning(f"hwp5txt failed for {path}: {e}, trying olefile fallback")
    
    try:
        import olefile
        if olefile.isOleFile(path):
            ole = olefile.OleFileIO(path)
            if ole.exists('PrvText'):
                stream = ole.openstream('PrvText')
                data = stream.read()
                text = data.decode('utf-16le', errors='ignore')
                ole.close()
                return text
    except Exception as e:
        logger.error(f"HWP extraction error {path}: {e}")
    
    return ""


def extract_image_ocr(path):
    """Extract text from image using pytesseract OCR."""
    try:
        from PIL import Image
        import pytesseract
        
        ocr_lang = os.getenv('OCR_LANG', 'kor+eng')
        img = Image.open(path)
        text = pytesseract.image_to_string(img, lang=ocr_lang)
        return text
    except Exception as e:
        logger.error(f"OCR extraction error {path}: {e}")
        return ""


def download_file(url, save_dir):
    """Download a file from URL and save to directory."""
    try:
        os.makedirs(save_dir, exist_ok=True)
        headers = {'User-Agent': USER_AGENT}
        r = httpx.get(url, headers=headers, timeout=30.0, follow_redirects=True)
        if r.status_code != 200:
            logger.warning(f"Failed to download {url}: status {r.status_code}")
            return None
        
        # Generate filename from URL
        parsed = urlparse(url)
        filename = os.path.basename(parsed.path)
        if not filename or '.' not in filename:
            # Guess extension from content-type
            ctype = r.headers.get('content-type', '').split(';')[0].strip()
            ext = mimetypes.guess_extension(ctype) or '.bin'
            filename = f"{hash(url)}{ext}"
        
        filepath = os.path.join(save_dir, filename)
        with open(filepath, 'wb') as f:
            f.write(r.content)
        
        logger.info(f"Downloaded: {url} -> {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Download error {url}: {e}")
        return None


def detect_attachments(html, base_url, page_url):
    """Detect and return metadata for attachments (images, PDFs, HWP, DOCX)."""
    soup = BeautifulSoup(html, 'lxml')
    attachments = []
    
    # Image attachments
    for img in soup.find_all('img', src=True):
        src = img['src']
        if not src:
            continue
        img_url = urljoin(page_url, src)
        if img_url.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
            attachments.append({
                'type': 'image',
                'url': img_url,
                'source_page': page_url
            })
    
    # Document attachments (PDF, HWP, DOC, DOCX)
    for a in soup.find_all('a', href=True):
        href = a['href']
        if not href:
            continue
        full_url = urljoin(page_url, href)
        lower_url = full_url.lower()
        
        if '.pdf' in lower_url:
            attachments.append({
                'type': 'pdf',
                'url': full_url,
                'source_page': page_url
            })
        elif '.hwp' in lower_url:
            attachments.append({
                'type': 'hwp',
                'url': full_url,
                'source_page': page_url
            })
        elif lower_url.endswith('.doc') or '.docx' in lower_url:
            attachments.append({
                'type': 'docx',
                'url': full_url,
                'source_page': page_url
            })
    
    return attachments

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
    """Attempt to extract fuller page content, not only paragraphs.

    Strategy:
    1. Remove script/style/noscript.
    2. Prefer <main>, <article> blocks if present; otherwise use body.
    3. Assemble text from block-level elements (p, h1-h6, li, td, th, pre, code).
    4. Fallback to full visible text if structured extraction too small.
    
    Returns: (title, text, attachments)
    """
    soup = BeautifulSoup(html, 'lxml')
    title = (soup.title.string.strip() if soup.title and soup.title.string else '')

    for tag in soup(['script', 'style', 'noscript']):
        tag.decompose()

    container = soup.find('main') or soup.find('article') or soup.body or soup
    parts = []
    selectors = ['h1','h2','h3','h4','h5','h6','p','li','td','th','pre','code']
    for sel in selectors:
        for el in container.find_all(sel):
            txt = el.get_text(separator=' ', strip=True)
            if txt:
                parts.append(txt)

    text = '\n\n'.join(parts)
    if len(text) < 400:
        # fallback: everything visible (may include nav/footer but ensures coverage)
        text = soup.get_text(separator=' ', strip=True)

    # normalize excessive whitespace
    import re
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Detect attachments
    attachments = detect_attachments(html, base_url, url)
    
    return title, text, attachments


def crawl(start_urls, allowlist, max_pages=300, max_depth=2, delay_ms=1500):
    seen = set()
    q = deque()
    for u in start_urls:
        q.append((u,0))

    results = []
    attachment_dir = os.getenv('ATTACHMENT_DIR', './data/attachments')
    
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
        title, text, attachments = extract_text_and_title(html, base, url)
        if not text or len(text) < 400:
            continue
        
        # Download attachments
        downloaded_attachments = []
        for att in attachments:
            filepath = download_file(att['url'], attachment_dir)
            if filepath:
                downloaded_attachments.append({
                    'type': att['type'],
                    'url': att['url'],
                    'path': filepath,
                    'source_page': att['source_page']
                })
        
        results.append({
            'url': url,
            'title': title,
            'text': text,
            'attachments': downloaded_attachments
        })

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
