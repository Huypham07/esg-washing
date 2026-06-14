"""Evidence-support score (spec 03 #4).

support(claim) = max_{e in top-k} P_entail(e, claim)   (FEVER-style aggregation)
grounded@theta voi theta in {0.5, 0.7, 0.9} - sweep, khong chon cung.
"""
from __future__ import annotations


def support_score(entail_probs) -> float:
    """Max P_entail tren cac evidence top-k. Khong co evidence -> 0.0."""
    vals = list(entail_probs)
    return float(max(vals)) if vals else 0.0


def grounded_flags(support: float, thresholds=(0.5, 0.7, 0.9)) -> dict:
    return {f"grounded@{th}": bool(support >= th) for th in thresholds}
