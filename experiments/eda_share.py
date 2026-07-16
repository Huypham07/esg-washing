"""Aggregate ESG disclosure share by pillar (E/S/G) across all bank-years.

Replaces the per-(bank,year) heatmap with a single 3-bar summary in the
scores-EDA template. Figure text in English. -> experiments/figures/share_pillars.png
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from esgwash.eda.style import PALETTE, SCORE_CMAP, apply_rcparams, style_axes


def mean_share() -> dict:
    frames = []
    for f in sorted(glob.glob("outputs/cti/*/*/pillar_shares.parquet")):
        parts = f.replace(os.sep, "/").split("/")
        df = pd.read_parquet(f)
        df["bank"], df["year"] = parts[2], int(parts[3])
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    piv = data.pivot_table(index=["bank", "year"], columns="pillar", values="share")
    return {p: float(piv[p].mean()) for p in ["env", "soc", "gov"]}


def main(out_dir: str = "experiments/figures") -> Path:
    apply_rcparams()
    m = mean_share()
    labels = ["Environmental (E)", "Social (S)", "Governance (G)"]
    vals = [m["env"], m["soc"], m["gov"]]
    colors = [SCORE_CMAP(0.15), SCORE_CMAP(0.55), SCORE_CMAP(0.9)]

    fig, ax = plt.subplots(figsize=(6, 4.6), facecolor=PALETTE["paper"])
    style_axes(ax, "ESG disclosure share by pillar",
               "Mean share of ESG commitments per pillar (45 bank-years).")
    bars = ax.bar(labels, vals, color=colors, edgecolor=PALETTE["paper"], width=0.62)
    ax.set_ylabel("Mean share")
    ax.set_ylim(0, max(vals) * 1.18)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", color=PALETTE["ink"], fontweight="bold", fontsize=11)
    fig.tight_layout()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "share_pillars.png"
    fig.savefig(p, bbox_inches="tight", facecolor=PALETTE["paper"])
    plt.close(fig)
    return p


if __name__ == "__main__":
    print(f"-> {main()}")
