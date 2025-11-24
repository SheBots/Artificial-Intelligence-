import os

def split_text(text: str, chunk_size=None, overlap=None):
    """Split text into overlapping chunks.

    If chunk_size / overlap not provided, read from env CHUNK_SIZE / CHUNK_OVERLAP.
    Defaults (when unset): chunk_size=1800 characters, overlap=250 characters.
    Ensures we cover full page content sequentially until end.
    """
    if chunk_size is None:
        try:
            chunk_size = int(os.getenv('CHUNK_SIZE', '1800'))
        except ValueError:
            chunk_size = 1800
    if overlap is None:
        try:
            overlap = int(os.getenv('CHUNK_OVERLAP', '250'))
        except ValueError:
            overlap = 250

    if chunk_size <= 0:
        return [text]
    if overlap < 0:
        overlap = 0

    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        # extend end to nearest whitespace to avoid cutting inside a word
        if end < length:
            while end < length and not text[end].isspace() and (end - start) < (chunk_size + 40):
                end += 1
            chunk = text[start:end]
        chunks.append(chunk)
        if end >= length:
            break
        start = max(0, end - overlap)
    return chunks
