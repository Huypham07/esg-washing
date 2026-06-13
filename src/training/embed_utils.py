from __future__ import annotations

from functools import lru_cache

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine


@lru_cache(maxsize=16384)
def encode_cached(encoder, text: str) -> np.ndarray:
    return encoder.encode(text, convert_to_numpy=True)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(sklearn_cosine(a.reshape(1, -1), b.reshape(1, -1))[0][0])
