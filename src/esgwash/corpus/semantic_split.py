"""Tach cau theo ranh gioi y (topic shift) roi goi xuong tran token.

Tieu chi CHINH = ranh gioi ngu nghia: cat giua 2 cau ke nhau khi cosine embedding
tut duoi nguong -> mot block da-y thanh nhieu don vi. Tran token chi la rao an toan
de bao ve encoder (PhoBERT 256): mot doan cung-y qua dai moi bi cat them tai ranh
gioi cau. Khong bao gio cat ngang 1 cau.

Ham thuan (nhan embeddings + token_counts dung san) -> test khong can load model.
"""
from __future__ import annotations

import numpy as np


def adjacent_cosine(embeddings: np.ndarray) -> np.ndarray:
    """Cosine giua cac cap cau ke nhau -> mang do dai n-1 (n = so cau)."""
    if len(embeddings) < 2:
        return np.empty(0)
    a = embeddings[:-1]
    b = embeddings[1:]
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    denom = np.where((na * nb) == 0, 1.0, na * nb)
    return np.sum(a * b, axis=1) / denom


def segment_indices(sims: np.ndarray, threshold: float) -> list[list[int]]:
    """Gom index cau lien tiep thanh doan; cat sau cau i khi sims[i] < threshold."""
    segs, cur = [], [0]
    for i, s in enumerate(sims):
        if s < threshold:
            segs.append(cur)
            cur = [i + 1]
        else:
            cur.append(i + 1)
    segs.append(cur)
    return segs


def pack_to_token_cap(sentences: list[str], token_counts: list[int],
                      max_tokens: int) -> list[list[str]]:
    """Goi cac cau (cung 1 doan y) thanh don vi <= max_tokens, khong cat ngang cau.
    Mot cau don > max_tokens duoc giu rieng (encoder se truncate cau do, chap nhan)."""
    units, cur, cur_tok = [], [], 0
    for s, t in zip(sentences, token_counts):
        if cur and cur_tok + t > max_tokens:
            units.append(cur)
            cur, cur_tok = [], 0
        cur.append(s)
        cur_tok += t
    if cur:
        units.append(cur)
    return units


def semantic_units(sentences: list[str], embeddings: np.ndarray,
                   token_counts: list[int], threshold: float,
                   max_tokens: int) -> list[list[str]]:
    """Tach y -> goi token cap. Tra ve list cac don vi (moi don vi = list cau)."""
    if not sentences:
        return []
    sims = adjacent_cosine(np.asarray(embeddings, dtype=float))
    out = []
    for seg in segment_indices(sims, threshold):
        seg_sents = [sentences[i] for i in seg]
        seg_toks = [token_counts[i] for i in seg]
        out.extend(pack_to_token_cap(seg_sents, seg_toks, max_tokens))
    return out
