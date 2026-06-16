"""Lay mau audit tay cho thang specificity (spec 2026-06-16 §5).

- sample_for_audit: rut n don vi/tru ra de gan tay Muc 0/1/2 (cot gold_level de trong),
  do agreement giua pipeline va nguoi -> chung minh CTI/QDR dang tin.
- audit_agreement: accuracy spec_level (pipeline) vs gold_level (nguoi) tren dong da gan.
"""
from __future__ import annotations

import pandas as pd

AUDIT_COLS = ["bank", "year", "pillar", "chunk_index", "content_text", "spec_level"]


def sample_for_audit(long: pd.DataFrame, n_per_pillar: int = 100,
                     seed: int = 42) -> pd.DataFrame:
    """Mau can bang theo tru de gan tay Muc 0/1/2 (cot gold_level de trong)."""
    parts = [g.sample(min(n_per_pillar, len(g)), random_state=seed)
             for _, g in long.groupby("pillar")]
    out = pd.concat(parts, ignore_index=True)
    out = out[[c for c in AUDIT_COLS if c in out.columns]].copy()
    out["gold_level"] = ""        # nguoi gan tay 0/1/2
    return out.reset_index(drop=True)


def audit_agreement(audited: pd.DataFrame) -> float:
    """Accuracy spec_level == gold_level (chi tren dong da gan)."""
    a = audited.dropna(subset=["gold_level"])
    a = a[a["gold_level"].astype(str).str.len() > 0]
    if a.empty:
        return float("nan")
    return float((a["spec_level"].astype(int) == a["gold_level"].astype(int)).mean())
