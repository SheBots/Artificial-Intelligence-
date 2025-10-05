import faiss
import numpy as np
import os, json
from typing import List

class Doc:
    def __init__(self, text, meta):
        self.text = text
        self.meta = meta


class FaissStore:
    def __init__(self, dim, index_path, docstore_path):
        self.dim = dim
        self.index_path = index_path
        self.docstore_path = docstore_path
        self.index = None
        self.docstore = []

    def load_or_create(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = faiss.IndexFlatIP(self.dim)
        if os.path.exists(self.docstore_path):
            with open(self.docstore_path,'r',encoding='utf-8') as f:
                for line in f:
                    self.docstore.append(json.loads(line))

    def persist(self):
        os.makedirs(os.path.dirname(self.index_path) or '.', exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.docstore_path,'w',encoding='utf-8') as f:
            for d in self.docstore:
                f.write(json.dumps(d, ensure_ascii=False)+'\n')

    def upsert(self, embeddings: List[List[float]], docs: List[Doc]):
        vecs = np.array(embeddings).astype('float32')
        if self.index.ntotal == 0:
            # convert to inner product by normalizing
            faiss.normalize_L2(vecs)
            self.index.add(vecs)
        else:
            faiss.normalize_L2(vecs)
            self.index.add(vecs)
        for d in docs:
            self.docstore.append({'text': d.text, **d.meta})

    def search(self, query_emb, k=5):
        import numpy as np
        vec = np.array([query_emb]).astype('float32')
        faiss.normalize_L2(vec)
        D, I = self.index.search(vec, k)
        results = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx < 0 or idx >= len(self.docstore):
                continue
            doc = self.docstore[idx]
            results.append({'text': doc.get('text'), 'score': float(score), 'url': doc.get('url'), 'title': doc.get('title')})
        return results
