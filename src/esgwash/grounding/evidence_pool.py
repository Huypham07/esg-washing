"""Pool ứng viên bằng chứng (spec 03 #1) — thu hẹp, không lấy cả corpus.

Đơn vị = chunk. Ứng viên: cùng doc_id và (có số liệu | specific-fact
is_specific=1 & is_commitment=0). Loại chính chunk claim (xử lý ở ground_claims).
"""
from __future__ import annotations

import re

import pandas as pd

# so + (don vi pho bien trong BCTN/ESG)
NUMERIC_PATTERN = re.compile(
    r"\d[\d.,]*\s*(?:%|tỷ|triệu|nghìn|tấn|t[aấ]n\s*CO2|CO2|CO₂|MWh|kWh|GWh|ha|km|"
    r"tỷ\s*đồng|triệu\s*đồng|đồng|VND|USD|m3|m³|kg)",
    re.IGNORECASE)


def candidate_mask(doc_chunks: pd.DataFrame, config: dict | None = None) -> pd.Series:
    """Boolean mask ung vien bang chung cho ca doc (tinh 1 lan): regex so lieu + specific-fact."""
    cfg = config or {}
    pool_cfg = cfg.get("pool", {})
    m = pd.Series(False, index=doc_chunks.index)
    if pool_cfg.get("numeric_regex", True):
        m |= doc_chunks["content_text"].astype(str).str.contains(NUMERIC_PATTERN)
    if pool_cfg.get("specific_fact", True) and {"is_specific", "is_commitment"} <= set(doc_chunks):
        m |= (doc_chunks["is_specific"] == 1) & (doc_chunks["is_commitment"] == 0)
    return m


# --- Neo cho pool Mức 1 (hành động có tên — pool KHÔNG lọc số) ---
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_VI_STOP = {"và", "của", "các", "có", "được", "cho", "trong", "với", "đã", "đang", "sẽ", "là",
            "những", "một", "này", "đó", "theo", "để", "khi", "từ", "tại", "về", "như", "hoặc",
            "cũng", "còn", "trên", "dưới", "đến", "bằng", "nhằm", "việc", "công", "ngân", "hàng"}


def action_anchors(action: str, min_len: int = 4, top: int = 8) -> list[str]:
    """Neo overlap cho pool L1: content word (lowercase, len>=min_len, bỏ stopword); giữ thứ tự + unique."""
    seen: set[str] = set()
    out: list[str] = []
    for t in _WORD_RE.findall(str(action).lower()):
        if len(t) >= min_len and t not in _VI_STOP and t not in seen:
            seen.add(t)
            out.append(t)
    return out[:top]


def anchor_overlap(sentence: str, anchors: list[str]) -> bool:
    """Câu chia sẻ >=1 neo (substring lowercase) — thu hẹp pool L1 quanh hành động được nhắc lại."""
    if not anchors:
        return False
    s = str(sentence).lower()
    return any(a in s for a in anchors)
