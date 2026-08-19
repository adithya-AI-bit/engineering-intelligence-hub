import json
import numpy as np
from dataclasses import asdict
from hub.chunking import Chunk
from hub.config import OPENAI_EMBEDDER


class VectorIndex:
    def __init__(self, embedder=OPENAI_EMBEDDER):
        self.chunks = []
        self.vectors = None
        self.embedder = embedder

    def add(self, chunks, vectors):
        if len(chunks) != len(vectors):
            raise ValueError(f"got {len(chunks)} chunks but {len(vectors)} vectors")
        self.chunks.extend(chunks)
        self.vectors = vectors if self.vectors is None else np.vstack([self.vectors, vectors])

    def search(self, query_vector, k):
        if self.vectors is None or not self.chunks:
            return []

        query = np.asarray(query_vector, dtype=np.float32)
        if query.shape[0] != self.vectors.shape[1]:
            raise ValueError(
                f"query has {query.shape[0]} dimensions but this index has "
                f"{self.vectors.shape[1]}. It was built with the '{self.embedder}' embedder; "
                f"query it with the same one, or rebuild it with ingest.py."
            )
        query = query / max(float(np.linalg.norm(query)), 1e-9)
        scores = self.vectors @ query
        top = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top]

    def save(self, path):
        np.savez(
            path,
            vectors=self.vectors if self.vectors is not None else np.zeros((0, 0), dtype=np.float32),
            chunks=np.array(json.dumps([asdict(c) for c in self.chunks])),
            embedder=np.array(self.embedder),
        )

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)
        # Indexes written before the embedder tag existed are assumed to be OpenAI.
        index = cls(embedder=str(data["embedder"]) if "embedder" in data else OPENAI_EMBEDDER)
        index.chunks = [Chunk(**c) for c in json.loads(str(data["chunks"]))]
        index.vectors = data["vectors"] if index.chunks else None
        return index
