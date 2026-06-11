"""Lam sach van ban OCR (spec 01 #1): NFC, sua loi OCR, loc cau rac, dedup.

KHONG xoa so lieu/don vi - chung la tin hieu specificity & evidence pool.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata

import numpy as np

PAGE_NUM_RE = re.compile(r"^\s*(?:trang\s*)?\d{1,4}\s*(?:/\s*\d{1,4})?\s*$", re.IGNORECASE)
IMAGE_TAG_RE = re.compile(r"<!--\s*image\s*-->|<image[^>]*>", re.IGNORECASE)
HYPHEN_BREAK_RE = re.compile(r"(\w)-\s*\n\s*(\w)")
LETTER_RE = re.compile(r"[a-zA-ZÀ-ỹ]")
WORD_RE = re.compile(r"[a-zA-ZÀ-ỹ]{2,}")

OCR_CHAR_FIXES = str.maketrans({"−": "-", "–": "-", "—": "-", "�": "", "­": ""})


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text).translate(OCR_CHAR_FIXES)


def clean_raw_text(text: str) -> str:
    text = normalize_unicode(text)
    text = HYPHEN_BREAK_RE.sub(r"\1\2", text)
    return IMAGE_TAG_RE.sub("", text)


def is_noise_line(line: str) -> bool:
    line = line.strip()
    return (not line) or bool(PAGE_NUM_RE.match(line)) or not LETTER_RE.search(line)


def is_valid_sentence(sentence: str, min_chars: int = 10, min_words: int = 3) -> bool:
    sentence = sentence.strip()
    return len(sentence) >= min_chars and len(WORD_RE.findall(sentence)) >= min_words


def _shingles(text: str, k: int = 4) -> set:
    toks = re.findall(r"\w+", text.lower())
    if len(toks) < k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)}


def _minhash_sig(shingles: set, perms: np.ndarray) -> np.ndarray:
    hs = np.array([int(hashlib.md5(s.encode()).hexdigest()[:15], 16) for s in shingles],
                  dtype=np.uint64)
    prime = np.uint64((1 << 61) - 1)
    return ((perms[:, :1] * hs + perms[:, 1:2]) % prime).min(axis=1)


def dedup_mask(texts: list[str], near_dup: bool = True, num_perm: int = 64,
               threshold: float = 0.85, seed: int = 42) -> np.ndarray:
    """Mask bool (True = giu): exact dedup + minhash LSH near-dup. Scope do caller quyet."""
    keep = np.ones(len(texts), dtype=bool)
    seen: set[str] = set()
    rng = np.random.default_rng(seed)
    perms = rng.integers(1, (1 << 61) - 1, size=(num_perm, 2), dtype=np.uint64)
    bands, rows = 16, num_perm // 16
    sigs: dict[int, np.ndarray] = {}
    buckets: dict[bytes, list[int]] = {}

    for i, t in enumerate(texts):
        key = re.sub(r"\s+", " ", t.lower()).strip()
        if key in seen:
            keep[i] = False
            continue
        seen.add(key)
        if not near_dup:
            continue
        sh = _shingles(t)
        if not sh:
            continue
        sig = _minhash_sig(sh, perms)
        band_keys = [bytes([b]) + sig[b * rows:(b + 1) * rows].tobytes() for b in range(bands)]
        cands = {j for bk in band_keys for j in buckets.get(bk, [])}
        if any((sigs[j] == sig).mean() >= threshold for j in cands):
            keep[i] = False
            continue
        sigs[i] = sig
        for bk in band_keys:
            buckets.setdefault(bk, []).append(i)
    return keep


def dedup_per_doc(df, text_col: str = "sentence", doc_col: str = "doc_id",
                  near_dup: bool = True):
    """Exact + near-dup trong cung doc_id; tra ve df da loc."""
    keep = np.ones(len(df), dtype=bool)
    for _, idx in df.groupby(doc_col, sort=False).groups.items():
        sub = df.loc[idx, text_col].tolist()
        keep[df.index.get_indexer(idx)] = dedup_mask(sub, near_dup=near_dup)
    return df[keep].reset_index(drop=True)
