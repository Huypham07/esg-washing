"""Shared figure style (scores-EDA notebook look). All figure text in English."""
from __future__ import annotations

import numpy as np
from matplotlib import colors as mcolors

PALETTE = {
    "paper": "#f7f4ed",
    "panel": "#efe8da",
    "ink": "#1f2a33",
    "muted": "#5c6770",
    "grid": "#c9bfa9",
    "accent": "#b55239",
    "accent2": "#2b7a78",
    "highlight": "#d6a84f",
}

SCORE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "score_map", ["#113b5c", "#3b7a57", "#d6a84f", "#b55239"])
SCORE_CMAP.set_bad(PALETTE["panel"])

DELTA_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "delta_map", [PALETTE["accent"], "#f7f4ed", PALETTE["accent2"]])
DELTA_NORM = mcolors.TwoSlopeNorm(vmin=-0.9, vcenter=0.0, vmax=0.9)


def apply_rcparams() -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.figsize": (14, 8), "figure.dpi": 140,
        "axes.titlesize": 16, "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "font.family": "DejaVu Sans",
    })


def style_axes(ax, title=None, subtitle=None, title_pad=28):
    ax.set_facecolor(PALETTE["panel"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.9, alpha=0.45)
    ax.tick_params(colors=PALETTE["ink"])
    if title is not None:
        ax.set_title(title, loc="left", color=PALETTE["ink"], pad=title_pad, fontweight="bold")
    if subtitle is not None:
        ax.text(0.0, 1.0, subtitle, transform=ax.transAxes, fontsize=10,
                color=PALETTE["muted"], va="bottom", ha="left")
    return ax


def shannon_entropy_bits(probs) -> float:
    """-Σ p log2 p over positive entries; accepts counts or probabilities."""
    p = np.asarray(list(probs), dtype=float)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p[p > 0] / s
    return float(-(p * np.log2(p)).sum())


def effective_states(probs) -> float:
    return float(2 ** shannon_entropy_bits(probs))
