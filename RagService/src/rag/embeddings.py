# Ensure huggingface_hub provides cached_download (older sentence-transformers expects it)
try:
    import huggingface_hub as _hf
    if not hasattr(_hf, 'cached_download') and hasattr(_hf, 'hf_hub_download'):
        setattr(_hf, 'cached_download', getattr(_hf, 'hf_hub_download'))
except Exception:
    # ignore if huggingface_hub isn't importable yet; sentence-transformers will show an error later
    _hf = None

from sentence_transformers import SentenceTransformer
import os

_model = None

def get_embedding_model(name=None):
    global _model
    if _model is None:
        model_name = name or os.getenv('EMBEDDING_MODEL','sentence-transformers/all-MiniLM-L6-v2')
        _model = SentenceTransformer(model_name)
    return _model

def embed_texts(texts, model=None):
    m = get_embedding_model(model)
    return m.encode(texts, show_progress_bar=False)
