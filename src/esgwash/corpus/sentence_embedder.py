"""Embed cau tieng Viet bang bi-encoder (cho tach ngu nghia khi build chunk).

bkai vietnamese-bi-encoder dua tren PhoBERT nen can word-seg dau vao
(dung chung segmentation.word_segment_batch). Tach rieng de build_chunks khong
phu thuoc package grounding (da chuyen legacy)."""
from __future__ import annotations

import numpy as np

from esgwash.nlp.segmentation import word_segment_batch

DEFAULT_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"


class SentenceEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        seg = word_segment_batch([str(t) for t in texts])
        return self.model.encode(seg, normalize_embeddings=True, show_progress_bar=False)
