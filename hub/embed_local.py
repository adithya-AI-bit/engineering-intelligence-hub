"""Offline embedder: no API key, no network, no extra dependencies.

Lexical rather than semantic, so it will not match paraphrases the way
text-embedding-3-small does. It exists so the pipeline and the eval harness can
be run and demonstrated without an OpenAI account.
"""
import math
import re
import zlib
import numpy as np

LOCAL_DIM = 512
TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")


def _tokens(text):
    for raw in TOKEN_PATTERN.findall(text.lower()):
        yield raw
        # split snake_case and run-on identifiers so retry_policy matches "retry"
        for part in raw.split("_"):
            if part and part != raw:
                yield part


def embed_texts_local(texts, dim=LOCAL_DIM):
    matrix = np.zeros((len(texts), dim), dtype=np.float32)

    for row, text in enumerate(texts):
        counts = {}
        for token in _tokens(text):
            # zlib.crc32, not the built-in hash(): string hashing is randomised
            # per process, so a saved index would not match queries on reload.
            bucket = zlib.crc32(token.encode()) % dim
            counts[bucket] = counts.get(bucket, 0) + 1
        for bucket, count in counts.items():
            matrix[row, bucket] = 1.0 + math.log(count)

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-9, None)
