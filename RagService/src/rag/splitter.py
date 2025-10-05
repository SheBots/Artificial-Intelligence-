def split_text(text: str, chunk_size=700, overlap=120):
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = max(0, end - overlap)
        if end >= length:
            break
    return chunks
