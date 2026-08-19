import numpy as np
from hub.config import EMBED_MODEL, EMBED_BATCH_SIZE


def embed_texts(texts, client=None):
    if client is None:
        from openai import OpenAI
        client = OpenAI(timeout=60.0, max_retries=2)

    vectors = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start:start + EMBED_BATCH_SIZE]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend(item.embedding for item in response.data)

    matrix = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-9, None)
