"""Candidate evidence pool (spec 03 #1) - thu hep, khong lay ca corpus.

Ung vien: cung doc_id, va (co so lieu | block_type table/list | specific-fact
is_specific=1 & is_commitment=0). Loai chinh claim va cac cau cung block.
"""
from __future__ import annotations

import re

import pandas as pd

# so + (don vi pho bien trong BCTN/ESG)
NUMERIC_PATTERN = re.compile(
    r"\d[\d.,]*\s*(?:%|tỷ|triệu|nghìn|tấn|t[aấ]n\s*CO2|CO2|CO₂|MWh|kWh|GWh|ha|km|"
    r"tỷ\s*đồng|triệu\s*đồng|đồng|VND|USD|m3|m³|kg)",
    re.IGNORECASE)


def candidate_mask(doc_sentences: pd.DataFrame, config: dict | None = None) -> pd.Series:
    """Boolean mask cho ca doc (tinh 1 lan). Vectorize regex + block_type + specific-fact."""
    cfg = config or {}
    pool_cfg = cfg.get("pool", {})
    m = pd.Series(False, index=doc_sentences.index)
    if pool_cfg.get("numeric_regex", True):
        m |= doc_sentences["sentence"].astype(str).str.contains(NUMERIC_PATTERN)
    bts = pool_cfg.get("block_types", ["table", "list"])
    if bts and "block_type" in doc_sentences:
        m |= doc_sentences["block_type"].isin(bts)
    if pool_cfg.get("specific_fact", True) and {"is_specific", "is_commitment"} <= set(doc_sentences):
        m |= (doc_sentences["is_specific"] == 1) & (doc_sentences["is_commitment"] == 0)
    return m


def build_pool(claim_row, doc_candidates: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """doc_candidates: cau ung vien cua CUNG doc_id (da loc candidate_mask).
    Loai chinh claim + cac cau cung block (tranh tu do chinh minh)."""
    pool = doc_candidates[doc_candidates["sent_id"] != claim_row["sent_id"]]
    if "block_id" in pool:
        pool = pool[pool["block_id"] != claim_row["block_id"]]
    return pool
