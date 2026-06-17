"""Điểm evidence-support (spec 03 #4).

support(claim) = max_{e in top-k} P_entail(e, claim)   (gộp kiểu FEVER)
grounded@theta với theta in {0.5, 0.7, 0.9} — quét ngưỡng, không chọn cứng.
"""
from __future__ import annotations


def support_score(entail_probs) -> float:
    """Max P_entail tren cac evidence top-k. Khong co evidence -> 0.0."""
    vals = list(entail_probs)
    return float(max(vals)) if vals else 0.0


def support_score_l1(entail_probs, contra_probs, lam: float = 0.5) -> float:
    """Support corroboration Muc 1: max tren top-k cua (P_entail - lam*P_contra).

    Phat contradiction de cau phan chung khong bi tinh la do. 0.0 neu khong co evidence."""
    pairs = list(zip(entail_probs, contra_probs))
    if not pairs:
        return 0.0
    return float(max(e - lam * c for e, c in pairs))


def grounded_flags(support: float, thresholds=(0.5, 0.7, 0.9)) -> dict:
    return {f"grounded@{th}": bool(support >= th) for th in thresholds}
