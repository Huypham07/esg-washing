"""Lam sach van ban OCR (spec 01 #1): NFC, sua loi OCR, loc cau rac, dedup.

KHONG xoa so lieu/don vi - chung la tin hieu specificity & evidence pool.
"""


def normalize_unicode(text: str) -> str:
    """NFC + chuan hoa U+2212 (minus), U+FFFD (luu y trong README en_gold)."""
    raise NotImplementedError


def is_noise_sentence(sentence: str, section_title: str | None = None) -> bool:
    """Header/footer, so trang, muc luc, cau < min_len. Tai dung heuristics code cu."""
    raise NotImplementedError


def dedup_per_doc(df, near_dup: bool = True):
    """Exact + minhash near-dup, pham vi trong cung doc_id."""
    raise NotImplementedError
