import re
from bs4 import BeautifulSoup

def clean_text(text: str) -> str:
    # strip excessive whitespace and scripts
    soup = BeautifulSoup(text, 'lxml')
    for s in soup(['script','style','noscript']):
        s.decompose()
    t = soup.get_text(separator=' ', strip=True)
    t = re.sub(r"\s+", ' ', t)
    return t.strip()
