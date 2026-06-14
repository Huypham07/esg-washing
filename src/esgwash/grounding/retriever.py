"""Retrieval top-k per doc bang bkai vietnamese-bi-encoder (spec 03 #2).

Cosine top-k=5, san sim >= 0.5. Pool nho per-doc -> khong can FAISS.
bkai dua tren PhoBERT -> word-segment input (dung chung VnCoreNLP singleton).
"""
from __future__ import annotations

import numpy as np

from esgwash.nlp.segmentation import word_segment_batch


class EvidenceRetriever:
    def __init__(self, config: dict):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            config.get("retriever", "bkai-foundation-models/vietnamese-bi-encoder"))
        self.k = config.get("top_k", 5)
        self.floor = config.get("sim_floor", 0.5)

    def embed(self, texts: list[str]) -> np.ndarray:
        seg = word_segment_batch([str(t) for t in texts])
        return self.model.encode(seg, normalize_embeddings=True, show_progress_bar=False)

    def topk(self, claim_vec: np.ndarray, pool_mat: np.ndarray, k: int | None = None):
        """-> (idx top-k vuot san, sim tuong ung). Cosine = dot vi da normalize."""
        k = k or self.k
        if len(pool_mat) == 0:
            return np.array([], dtype=int), np.array([])
        sims = pool_mat @ claim_vec
        order = np.argsort(-sims)[:k]
        keep = order[sims[order] >= self.floor]
        return keep, sims[keep]
