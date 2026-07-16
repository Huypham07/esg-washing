"""V3 sensitivity (spec 2026-06-16 §5): do on dinh ranking CTI giua cac ngan hang.

Grounding theta da bo, thay bang hai phep nhieu chay duoc tren output specificity:
- bootstrap_ranking_stability: resample chunk commitment-ESG trong moi (bank, year),
  tinh lai CTI cap ngan hang, do Kendall tau ranking vs ranking goc.
- leave_one_year_out: bo tung nam, do tau ranking ngan hang vs dung ca panel.
Ranking on dinh => CTI dung de xep hang ngan hang dang tin.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

_ESG = ("is_env", "is_soc", "is_gov")


def _commit_esg(classified: pd.DataFrame) -> pd.DataFrame:
    """Cam ket co it nhat 1 tru ESG duong (cong topic) — denominator cua CTI."""
    esg = classified[list(_ESG)].max(axis=1).astype(bool)
    return classified[(classified["is_commitment"] == 1) & esg]


def _bank_cti(commit: pd.DataFrame) -> pd.Series:
    """CTI cap ngan hang = ti le spec_level==0, gop moi nam."""
    return commit.groupby("bank")["spec_level"].apply(lambda s: float((s == 0).mean()))


def bootstrap_ranking_stability(classified: pd.DataFrame, n_resamples: int = 1000,
                                seed: int = 42) -> dict:
    """Resample co hoan lai cam ket moi ngan hang -> Kendall tau ranking vs goc."""
    commit = _commit_esg(classified)
    base = _bank_cti(commit)
    base_rank = base.rank()
    base_top = base.idxmax()
    groups = {b: g["spec_level"].to_numpy(float) for b, g in commit.groupby("bank")}

    rng = np.random.default_rng(seed)
    taus, top1_hits = [], 0
    for _ in range(n_resamples):
        cti = {b: float((v[rng.integers(0, len(v), len(v))] == 0).mean())
               for b, v in groups.items()}
        s = pd.Series(cti)
        tau, _ = kendalltau(base_rank, s.rank())
        taus.append(tau)
        top1_hits += int(s.idxmax() == base_top)

    taus = np.asarray(taus, dtype=float)
    return {
        "n_banks": int(base.size),
        "base_ranking": base.sort_values(ascending=False).round(4).to_dict(),
        "kendall_tau_mean": round(float(np.nanmean(taus)), 4),
        "kendall_tau_p05": round(float(np.nanpercentile(taus, 5)), 4),
        "top1_retention": round(top1_hits / n_resamples, 4),
    }


def leave_one_year_out(classified: pd.DataFrame) -> dict:
    """Bo tung nam -> tau ranking ngan hang vs dung ca panel (do phu thuoc 1 nam)."""
    commit = _commit_esg(classified)
    base = _bank_cti(commit).rank()
    out = {}
    for y in sorted(commit["year"].unique()):
        r = _bank_cti(commit[commit["year"] != y]).rank()
        common = base.index.intersection(r.index)
        tau, _ = kendalltau(base[common], r[common])
        out[int(y)] = round(float(tau), 4)
    return {"per_dropped_year": out, "tau_min": round(min(out.values()), 4)}
