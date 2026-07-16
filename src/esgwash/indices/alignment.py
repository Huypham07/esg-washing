"""Embedding washing signal (RQ4): Boilerplate Reuse Index (BRI).

Assumes L2-normalized embeddings (cosine == dot product). Pure geometry:
cosine + max + mean, no learned weights.

BRI: per (bank,year), how similar each commitment chunk is to commitments of
OTHER banks in the same year. High BRI = recycled cross-bank language.

Mean-centering removes sentence-embedding anisotropy that otherwise compresses
all cosines into a high, uninformative band; it is applied corpus-wide before
any cosine. (SBS / claim-evidence backing was dropped: raw topical cosine does
not capture evidential backing — empirically null on full data.)
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


def center_normalize(emb: np.ndarray) -> np.ndarray:
    """Remove the corpus mean (anisotropy fix) then re-normalize to unit length."""
    emb = np.asarray(emb, dtype=float)
    if emb.shape[0] == 0:
        return emb
    c = emb - emb.mean(axis=0)
    n = np.linalg.norm(c, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return c / n


def boilerplate_reuse_index(emb: np.ndarray, banks: np.ndarray):
    """Per chunk: max cosine to chunks of OTHER banks. BRI = mean over chunks.
    Single-bank input -> (nan, zeros)."""
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
    """df = ESG-commitment chunks (cols bank, year) aligned row-wise with emb
    in positional order (emb[i] is df row i). emb is mean-centered corpus-wide
    before cosine. BRI uses same-year cross-bank comparison. -> bank, year, bri.
    """
    df = df.reset_index(drop=True)      # labels must equal positional emb rows
    emb = center_normalize(emb)
    rows = []
    for year, g_year in df.groupby("year"):
        idx_year = g_year.index.to_numpy()
        emb_year = emb[idx_year]
        banks_year = g_year["bank"].to_numpy()
        _, bri_per = boilerplate_reuse_index(emb_year, banks_year)
        bri_series = pd.Series(bri_per, index=idx_year)
        for bank, g in g_year.groupby("bank"):
            idx = g.index.to_numpy()
            rows.append({"bank": bank, "year": year,
                         "bri": float(bri_series.loc[idx].mean())})
    return pd.DataFrame(rows)
