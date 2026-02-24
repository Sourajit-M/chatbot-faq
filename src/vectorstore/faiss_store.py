import faiss
from typing import List, Dict
import numpy as np


class FAISSVectorStore:
    def __init__(self, embedding_dim: int):
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.documents: List[Dict] = []

    def add(self, embeddings: List[List[float]], documents: List[Dict]):
        vectors = np.array(embeddings).astype("float32")
        self.index.add(vectors)
        self.documents.extend(documents)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        query_vector = np.array([query_embedding]).astype("float32")
        _, indices = self.index.search(query_vector, top_k)

        results = []
        for idx in indices[0]:
            if idx < len(self.documents):
                results.append(self.documents[idx])

        return results