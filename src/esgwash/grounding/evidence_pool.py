"""Candidate evidence pool (spec 03 #1) - thu hep, khong lay ca corpus.

Don vi = CHUNK. Ung vien: cung doc_id, va (co so lieu | specific-fact
is_specific=1 & is_commitment=0). Loai chinh chunk claim (xu ly o ground_claims).
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
