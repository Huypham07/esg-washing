"""Percentile bootstrap CI (spec 04 #4): resample cau commitment trong o,
B=1000, CI 95%. Min-n: o |C| < 30 chi vao bang pooled (bank x pillar / bank x year)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def bootstrap_ci(values, statistic, n_resamples: int = 1000, ci: float = 0.95,
                 seed: int = 42) -> tuple[float, float, float]:
    """-> (diem uoc luong, lo, hi). Percentile bootstrap co hoan lai."""
    values = np.asarray(values, dtype=float)
    point = float(statistic(values)) if len(values) else float("nan")
    if len(values) < 2:
        return point, point, point
    rng = np.random.default_rng(seed)
    n = len(values)
    boots = np.array([statistic(values[rng.integers(0, n, n)]) for _ in range(n_resamples)])
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(boots, [alpha * 100, (1 - alpha) * 100])
    return point, float(lo), float(hi)


def pooled_cells(claims_long: pd.DataFrame, min_n: int = 30) -> pd.DataFrame:
    """Danh dau o (bank,year,pillar) du n hay khong (spec 04 #4): o |C| < min_n
    chi vao bang pooled. claims_long: long-format da loc is_commitment=1 hoac chua."""
    commit = claims_long[claims_long["is_commitment"] == 1]
    cnt = (commit.groupby(["bank", "year", "pillar"]).size()
           .rename("n_commit").reset_index())
    cnt["enough"] = cnt["n_commit"] >= min_n
    return cnt
