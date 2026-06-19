"""Embedding washing signals (RQ4). Assumes L2-normalized embeddings so
cosine == dot product. Pure geometry: cosine + max + mean, no learned weights.

SBS (Substance Backing Score): per (bank,year), how well VAGUE commitments
(spec_level=0) are backed by a nearby QUANTIFIED commitment (spec_level=2).
Low SBS = vague claims float free of evidence -> corroborates high CTI.

BRI (Boilerplate Reuse Index): how similar each commitment is to commitments
of OTHER banks. High BRI = recycled generic language = cheap-talk signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def pairwise_max_cosine(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.shape[0] == 0:
        return np.zeros(0)
    if B.shape[0] == 0:
        return np.zeros(A.shape[0])
    sims = A @ B.T                      # rows normalized -> cosine
    return sims.max(axis=1)


def substance_backing_score(emb: np.ndarray, spec_level: np.ndarray):
    emb = np.asarray(emb, dtype=float)
    spec_level = np.asarray(spec_level)
    vague = emb[spec_level == 0]
    quant = emb[spec_level == 2]
    if vague.shape[0] == 0:
        return float("nan"), np.zeros(0)
    if quant.shape[0] == 0:
        per = np.zeros(vague.shape[0])
        return 0.0, per
    per = pairwise_max_cosine(vague, quant)
    return float(per.mean()), per


def boilerplate_reuse_index(emb: np.ndarray, banks: np.ndarray):
    emb = np.asarray(emb, dtype=float)
    banks = np.asarray(banks)
    n = emb.shape[0]
    per = np.zeros(n)
    if np.unique(banks).size < 2:
        return float("nan"), per
    for i in range(n):
        other = emb[banks != banks[i]]
        per[i] = pairwise_max_cosine(emb[i:i + 1], other)[0]
    return float(per.mean()), per


def signals_per_panel(df: pd.DataFrame, emb: np.ndarray) -> pd.DataFrame:
    """df = ESG-commitment chunks (cols bank, year, spec_level) aligned row-wise
    with emb. BRI uses same-year cross-bank comparison. -> bank, year, sbs, bri."""
    rows = []
    for year, g_year in df.groupby("year"):
        idx_year = g_year.index.to_numpy()
        emb_year = emb[idx_year]
        banks_year = g_year["bank"].to_numpy()
        _, bri_per = boilerplate_reuse_index(emb_year, banks_year)
        bri_series = pd.Series(bri_per, index=idx_year)
        for bank, g in g_year.groupby("bank"):
            idx = g.index.to_numpy()
            sbs, _ = substance_backing_score(emb[idx], g["spec_level"].to_numpy())
            rows.append({"bank": bank, "year": year, "sbs": sbs,
                         "bri": float(bri_series.loc[idx].mean())})
    return pd.DataFrame(rows)
