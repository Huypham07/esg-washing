"""CTI & grounded-CTI (spec 04 #1-2; Bingler et al. 2022).

CTI(b,y,p)  = |{commitment & ~specific}| / |{commitment}|
gCTI(b,y,p) = |{commitment & (~specific | (specific & support<theta))}| / |{commitment}|
Chỉ là tỉ lệ output của classifier, không tự đặt trọng số.

Input `claims_long`: mỗi dòng = (câu commitment × trụ nó thuộc về), cột
bank, year, pillar, is_commitment, is_specific, [support].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from esgwash.indices.bootstrap import bootstrap_ci

CELL = ["bank", "year", "pillar"]


def _frac_nonspecific(v: np.ndarray) -> float:
    return float((v == 0).mean())


def compute_cti(claims_long: pd.DataFrame, n_resamples: int = 1000,
                ci: float = 0.95, seed: int = 42) -> pd.DataFrame:
    commit = claims_long[claims_long["is_commitment"] == 1]
    rows = []
    for (b, y, p), g in commit.groupby(CELL):
        spec = g["is_specific"].to_numpy(dtype=float)
        point, lo, hi = bootstrap_ci(spec, _frac_nonspecific, n_resamples, ci, seed)
        rows.append({"bank": b, "year": y, "pillar": p, "n_commit": len(g),
                     "cti": round(point, 4), "cti_lo": round(lo, 4), "cti_hi": round(hi, 4)})
    return pd.DataFrame(rows)


def compute_grounded_cti(claims_long: pd.DataFrame, theta: float,
                         n_resamples: int = 1000, ci: float = 0.95,
                         seed: int = 42) -> pd.DataFrame:
    """gCTI: cam ket "re" neu khong cu the, HOAC cu the nhung support<theta."""
    commit = claims_long[claims_long["is_commitment"] == 1].copy()
    support = commit.get("support", pd.Series(0.0, index=commit.index)).fillna(0.0)
    cheap = (commit["is_specific"] == 0) | ((commit["is_specific"] == 1) & (support < theta))
    commit["_cheap"] = cheap.astype(float)
    g_col = f"gcti@{theta}"
    rows = []
    for (b, y, p), g in commit.groupby(CELL):
        point, lo, hi = bootstrap_ci(g["_cheap"].to_numpy(), np.mean, n_resamples, ci, seed)
        rows.append({"bank": b, "year": y, "pillar": p, "n_commit": len(g),
                     g_col: round(point, 4), f"{g_col}_lo": round(lo, 4),
                     f"{g_col}_hi": round(hi, 4)})
    return pd.DataFrame(rows)


def build_cti_table(claims_long: pd.DataFrame, thetas=(0.5, 0.7, 0.9),
                    n_resamples: int = 1000, ci: float = 0.95,
                    seed: int = 42) -> pd.DataFrame:
    """Bang cti.parquet: CTI + gCTI@cac theta, merge theo o (bank,year,pillar)."""
    out = compute_cti(claims_long, n_resamples, ci, seed)
    for th in thetas:
        g = compute_grounded_cti(claims_long, th, n_resamples, ci, seed).drop(columns=["n_commit"])
        out = out.merge(g, on=CELL, how="left")
    return out
