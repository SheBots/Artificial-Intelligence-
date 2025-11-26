# clean.py — strict HTML / text cleaner for SheBots RAG

import re
from bs4 import BeautifulSoup


def clean_html_strict(html: str) -> str:
    """
    Aggressively clean KNU CSE HTML pages:
    - remove nav/header/footer/sidebars/forms/scripts/styles
    - keep only main content containers (FAQ body, article, board view, etc.)
    - strip repeated UI text (ENGLISH LOGIN, search bars, breadcrumbs)
    - normalize whitespace

    This is tuned for the structure of the CSE site so that RAG chunks contain
    mostly meaningful content instead of menus and layout text.
    """
    soup = BeautifulSoup(html, "lxml")

    # Remove global junk tags
    for tag in soup(
        [
            "nav",
            "header",
            "footer",
            "aside",
            "script",
            "style",
            "noscript",
            "form",
            "input",
            "button",
            "svg",
        ]
    ):
        tag.decompose()

    # Remove known KNU CSE UI blocks (menus, sidebars, breadcrumbs, etc.)
    for css in [
        ".gnb",
        ".lnb",
        ".snb",
        ".sub_nav",
        ".subMenu",
        ".topMenu",
        ".bottomMenu",
        ".breadcrumb",
        ".search",
        ".pagination",
        ".footer",
        ".header",
        ".sitemap",
        ".location",
        ".quick_menu",
    ]:
        for tag in soup.select(css):
            tag.decompose()

    # Prefer real content containers
    main = None
    for selector in [
        "main",
        "article",
        "#content",
        ".content",
        ".sub_content",
        ".write_view",
        ".board_view",
        ".view_cont",
        ".board",
        ".bbs_view",
    ]:
        main = soup.select_one(selector)
        if main:
            break

    container = main if main is not None else soup.body if soup.body else soup

    # Extract visible text
    text = container.get_text(separator=" ", strip=True)

    # Remove specific garbage phrases that often repeat on CSE site
    garbage_patterns = [
        r"ENGLISH LOGIN",
        r"English Login",
        r"사이트 내 전체검색",
        r"검색어 필수",
        r"검색하고자 하는 키워드 입력 후 Enter 또는 검색아이콘 클릭을 통해 검색해 주세요",
        r"검색하고자 하는 키워드 입력 후 Enter",
        r"통합검색은 홈페이지의 내용을 전체 검색합니다",
        r"사이트 맵",
        r"사이트맵",
        r"전체 메뉴",
        r"전체메뉴",
        r"닫기",
        r"열기",
        r"HOME\s*>\s*",
    ]
    for pat in garbage_patterns:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE)

    # Collapse very repetitive keywords that come from layout
    text = re.sub(r"(컴퓨터학부\s*){2,}", "컴퓨터학부 ", text)
    text = re.sub(r"(글솝\s*){2,}", "글솝 ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_text(text: str) -> str:
    """
    Backwards-compatible wrapper.

    For HTML we should already be passing the raw HTML into clean_html_strict().
    For plain text (PDF/DOCX/TXT) this still:
    - strips excessive whitespace
    - removes obvious garbage patterns if present
    """
    # Heuristic: if it looks like HTML, run through full HTML cleaner
    if "<html" in text.lower() or "<body" in text.lower() or "<head" in text.lower():
        return clean_html_strict(text)

    # Plain text: just normalize whitespace + remove a few known menu phrases
    cleaned = text
    for pat in [
        r"ENGLISH LOGIN",
        r"사이트 내 전체검색",
        r"검색어 필수",
        r"사이트 맵",
        r"사이트맵",
    ]:
        cleaned = re.sub(pat, " ", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned
