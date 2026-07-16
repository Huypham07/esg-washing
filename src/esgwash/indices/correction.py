"""Measurement-error correction cho VDR/NAR/QDR bang ACC (Adjusted Classify & Count).

Quantification-learning: classifier specificity co bias he thong (hay cham "mo ho" qua tay).
Goi M[i,j] = P(pred=i | true=j) uoc luong tu gold -> dao nguoc:  p_true = M^{-1} p_pred.
Clip am + renormalize (an toan khi M gan singular). Khong sua pipeline; hau-xu-ly tren p_pred co san.

M dung basis gold "relabel tach-co" (moi annotator cham 4 co doc lap -> luat goc suy muc),
da reconcile 07/07 khop chinh xac M demo cua spec + paper (QWK 0.66 / human 0.70).
"""
from __future__ import annotations

import numpy as np

N_LEVELS = 3


def build_M(model_levels, gold_levels) -> tuple[np.ndarray, np.ndarray]:
    """counts[pred, true] roi column-normalize -> M[i,j] = P(pred=i | true=j).

    model_levels, gold_levels: mang so nguyen 0/1/2 (chi cac dong gold-committed).
    Tra (M 3x3 float, counts 3x3 int)."""
    m = np.asarray(model_levels, dtype=int)
    g = np.asarray(gold_levels, dtype=int)
    counts = np.zeros((N_LEVELS, N_LEVELS), dtype=float)
    for pi, gj in zip(m, g):
        counts[pi, gj] += 1.0
    col = counts.sum(axis=0, keepdims=True)
    M = np.divide(counts, col, out=np.zeros_like(counts), where=col > 0)
    return M, counts.astype(int)


def acc_correct(p_pred, M) -> np.ndarray:
    """p_true = M^{-1} p_pred, clip ve >=0, renormalize sum=1. pinv neu M singular."""
    p_pred = np.asarray(p_pred, dtype=float)
    try:
        p_true = np.linalg.solve(M, p_pred)
    except np.linalg.LinAlgError:
        p_true = np.linalg.pinv(M) @ p_pred
    p_true = np.clip(p_true, 0.0, None)
    s = p_true.sum()
    return p_true / s if s > 0 else p_true


def correct_cells(cell_preds, M) -> np.ndarray:
    """Ap acc_correct cho tung dong (VDR,NAR,QDR) cua panel per-cell. Tra (n_cells, 3)."""
    return np.array([acc_correct(row, M) for row in np.asarray(cell_preds, dtype=float)])


def bootstrap_joint(model_levels, gold_levels, cell_preds,
                    n_boot: int = 2000, seed: int = 42, ci: float = 0.95):
    """Bootstrap KEP: moi replicate (a) resample gold -> rebuild M; (b) resample cell -> mean p_pred;
    invert -> corrected. Tra (point, lo, hi, n_skipped_singular, n_used).
    CI bao CA sampling (resample cell) VA measurement error (resample gold -> M dao dong)."""
    rng = np.random.default_rng(seed)
    m = np.asarray(model_levels, dtype=int)
    g = np.asarray(gold_levels, dtype=int)
    cells = np.asarray(cell_preds, dtype=float)
    ng, nc = len(m), len(cells)

    M0, _ = build_M(m, g)
    point = acc_correct(cells.mean(axis=0), M0)

    boots, skipped = [], 0
    for _ in range(n_boot):
        gi = rng.integers(0, ng, ng)
        Mb, _ = build_M(m[gi], g[gi])
        if np.any(Mb.sum(axis=0) == 0) or np.linalg.cond(Mb) > 1e12:
            skipped += 1
            continue
        ci_idx = rng.integers(0, nc, nc)
        boots.append(acc_correct(cells[ci_idx].mean(axis=0), Mb))
    boots = np.asarray(boots)
    alpha = (1 - ci) / 2
    lo = np.percentile(boots, alpha * 100, axis=0)
    hi = np.percentile(boots, (1 - alpha) * 100, axis=0)
    return point, lo, hi, skipped, len(boots)
