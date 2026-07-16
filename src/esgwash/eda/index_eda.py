"""Pure compute for index-level EDA (no plotting)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def washing_feature_matrix(panel: pd.DataFrame,
                           cols=("cti", "nar", "qdr", "n_commit")):
    cols = list(cols)
    rows = [f"{b} {y}" for b, y in zip(panel["bank"], panel["year"])]
    M = panel[cols].to_numpy(dtype=float)
    mu = M.mean(axis=0)
    sd = M.std(axis=0)  # ddof=0; PCA evr is a ratio so the (n-1)/n vs n scaling cancels
    sd[sd == 0] = 1.0
    X = (M - mu) / sd
    return X, rows, cols


def manual_pca(X: np.ndarray, n_components: int = 2):
    Xc = X - X.mean(axis=0)
    cov = np.cov(Xc, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)          # ascending
    order = np.argsort(vals)[::-1]            # descending
    vals, vecs = vals[order], vecs[:, order]
    loadings = vecs[:, :n_components]
    scores = Xc @ loadings
    total = vals.sum()
    evr = (vals[:n_components] / total) if total > 0 else np.zeros(n_components)
    return scores, loadings, evr


def trajectory_deltas(panel: pd.DataFrame, col: str = "cti") -> dict:
    out = {}
    for bank, g in panel.sort_values("year").groupby("bank"):
        vals = g[col].to_numpy(dtype=float).tolist()
        deltas = [0.0] + list(np.diff(vals))
        out[bank] = {"years": g["year"].tolist(), "values": vals, "deltas": deltas}
    return out


def quantile_ribbons(panel: pd.DataFrame, col: str = "cti") -> pd.DataFrame:
    g = panel.groupby("year")[col]
    rb = pd.DataFrame({
        "q10": g.quantile(0.10), "q25": g.quantile(0.25), "median": g.median(),
        "q75": g.quantile(0.75), "q90": g.quantile(0.90), "mean": g.mean()})
    return rb
