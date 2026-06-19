# experiments/eda_indices.py
"""Index-level EDA figures (English, scores-EDA style)."""
from __future__ import annotations

import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from esgwash.eda import index_eda as ie
from esgwash.eda.style import (DELTA_CMAP, DELTA_NORM, PALETTE, SCORE_CMAP,
                               apply_rcparams, style_axes)


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    fig.savefig(p, bbox_inches="tight", facecolor=PALETTE["paper"])
    plt.close(fig)
    return p


def _load_panel() -> pd.DataFrame:
    return pd.read_csv("experiments/panel/panel.csv")


def _load_shares() -> pd.DataFrame:
    files = sorted(glob.glob("outputs/cti/*/*/pillar_shares.parquet"))
    frames = [pd.read_parquet(f) for f in files]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fig_ribbons(panel, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), facecolor=PALETTE["paper"], sharex=True)
    for ax, col in zip(axes, ["cti", "nar", "qdr"]):
        rb = ie.quantile_ribbons(panel, col)
        yrs = rb.index.to_numpy()
        style_axes(ax, col.upper(), "Yearly inter-quantile band.")
        ax.fill_between(yrs, rb["q10"], rb["q90"], color=PALETTE["highlight"], alpha=0.18)
        ax.fill_between(yrs, rb["q25"], rb["q75"], color=PALETTE["highlight"], alpha=0.35)
        ax.plot(yrs, rb["mean"], color=PALETTE["accent"], linewidth=2.4, marker="o")
        ax.set_xticks(yrs)
        ax.set_ylim(0, 1.02)
    return _save(fig, out_dir, "index_ribbons.png")


def fig_cti_cartography(panel, out_dir):
    grid = panel.pivot_table(index="bank", columns="year", values="cti")
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=PALETTE["paper"])
    im = ax.imshow(np.ma.masked_invalid(grid.to_numpy()), aspect="auto",
                   cmap=SCORE_CMAP, vmin=0, vmax=1)
    ax.set_title("CTI cartography (bank x year)", loc="left", fontweight="bold",
                 color=PALETTE["ink"], pad=20)
    ax.set_xticks(range(len(grid.columns)), grid.columns)
    ax.set_yticks(range(len(grid.index)), grid.index)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="CTI")
    return _save(fig, out_dir, "index_cti_cartography.png")


def fig_cti_trajectories(panel, out_dir):
    tr = ie.trajectory_deltas(panel, "cti")
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=PALETTE["paper"])
    style_axes(ax, "CTI trajectories",
               "Segment colour = year-over-year change (red rising / teal falling).")
    for bank, d in tr.items():
        yrs, vals, deltas = d["years"], d["values"], d["deltas"]
        pts = np.array([yrs, vals]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap=DELTA_CMAP, norm=DELTA_NORM)
        lc.set_array(np.array(deltas[1:]))
        lc.set_linewidth(2.6)
        ax.add_collection(lc)
        ax.text(yrs[-1], vals[-1], f" {bank}", fontsize=8, color=PALETTE["ink"], va="center")
    ax.set_xlim(min(min(d["years"]) for d in tr.values()) - 0.2,
               max(max(d["years"]) for d in tr.values()) + 0.8)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Year")
    ax.set_ylabel("CTI")
    return _save(fig, out_dir, "index_cti_trajectories.png")


def fig_washing_pca(panel, out_dir):
    X, rows, feats = ie.washing_feature_matrix(panel)
    scores, loadings, evr = ie.manual_pca(X, 2)
    fig, ax = plt.subplots(figsize=(9, 7), facecolor=PALETTE["paper"])
    style_axes(ax, "Washing-space PCA (bank-year)",
               f"PC1 {evr[0]:.0%} / PC2 {evr[1]:.0%} of variance.")
    ax.scatter(scores[:, 0], scores[:, 1], s=80, c=PALETTE["accent"],
               edgecolor=PALETTE["paper"], zorder=3)
    for (x, y), lab in zip(scores, rows):
        ax.text(x, y, f" {lab}", fontsize=7, color=PALETTE["ink"])
    for i, f in enumerate(feats):
        ax.annotate(f, xy=(loadings[i, 0] * 3, loadings[i, 1] * 3),
                    color=PALETTE["accent2"], fontsize=9,
                    arrowprops=dict(arrowstyle="->", color=PALETTE["accent2"]),
                    xytext=(0, 0), textcoords="offset points")
    ax.axhline(0, color=PALETTE["grid"], linewidth=0.8)
    ax.axvline(0, color=PALETTE["grid"], linewidth=0.8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    return _save(fig, out_dir, "index_washing_pca.png")


def fig_selective_disclosure(shares, out_dir):
    grid = shares.pivot_table(index=["bank", "year"], columns="pillar", values="share_dev")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(grid))), facecolor=PALETTE["paper"])
    im = ax.imshow(grid.to_numpy(), aspect="auto", cmap=DELTA_CMAP, norm=DELTA_NORM)
    ax.set_title("Selective disclosure (share deviation from industry)", loc="left",
                 fontweight="bold", color=PALETTE["ink"], pad=20)
    ax.set_xticks(range(len(grid.columns)), grid.columns)
    ax.set_yticks(range(len(grid.index)), [f"{b} {y}" for b, y in grid.index], fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="share - industry share")
    return _save(fig, out_dir, "index_selective_disclosure.png")


def main(out_dir: str = "experiments/figures", panel=None, shares=None) -> list:
    apply_rcparams()
    if panel is None:
        panel = _load_panel()
    if shares is None:
        shares = _load_shares()
    out = Path(out_dir)
    return [fig_ribbons(panel, out), fig_cti_cartography(panel, out),
            fig_cti_trajectories(panel, out), fig_washing_pca(panel, out),
            fig_selective_disclosure(shares, out)]


if __name__ == "__main__":
    for p in main():
        print(f"-> {p}")
