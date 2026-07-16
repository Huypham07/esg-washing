# experiments/eda_corpus.py
"""Corpus-level EDA figures (English, scores-EDA style). Reads classified.parquet."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from esgwash.eda import corpus_eda as ce
from esgwash.eda.style import (PALETTE, SCORE_CMAP, apply_rcparams, style_axes)


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    fig.savefig(p, bbox_inches="tight", facecolor=PALETTE["paper"])
    plt.close(fig)
    return p


def fig_token_lattice(clf, out_dir: Path) -> Path:
    d = ce.token_distribution(clf)
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=PALETTE["paper"])
    style_axes(ax, "Chunk token-length distribution",
               "Per-chunk token counts; dashed lines mark P10/Q1/Median/Q3/P90.")
    vals = clf["token_count"].astype(float)
    ax.hist(vals, bins=40, color=PALETTE["accent2"], edgecolor=PALETTE["paper"])
    ymax = ax.get_ylim()[1]
    for q, lab in [(d["p10"], "P10"), (d["q1"], "Q1"), (d["median"], "Median"),
                   (d["q3"], "Q3"), (d["p90"], "P90")]:
        ax.axvline(q, color=PALETTE["ink"], linestyle=(0, (3, 3)), linewidth=1.0, alpha=0.35)
        ax.text(q, ymax * 0.97, lab, rotation=90, va="top", ha="center",
                fontsize=8, color=PALETTE["muted"])
    ax.set_xlabel("Tokens per chunk")
    ax.set_ylabel("Chunk count")
    return _save(fig, out_dir, "corpus_token_lattice.png")


def fig_coverage(clf, out_dir: Path) -> Path:
    cov = ce.coverage_matrix(clf, value="commitment")
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=PALETTE["paper"])
    im = ax.imshow(cov.to_numpy(), aspect="auto", cmap=SCORE_CMAP)
    ax.set_title("Commitment-chunk coverage (bank x year)", loc="left",
                 fontweight="bold", color=PALETTE["ink"], pad=20)
    ax.set_xticks(range(len(cov.columns)), cov.columns)
    ax.set_yticks(range(len(cov.index)), cov.index)
    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            ax.text(j, i, int(cov.iat[i, j]), ha="center", va="center",
                    fontsize=8, color=PALETTE["paper"])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Commitment chunks")
    return _save(fig, out_dir, "corpus_coverage.png")


def fig_labels(clf, out_dir: Path) -> Path:
    rates = ce.label_positive_rates(clf)
    spec = ce.spec_level_distribution(clf)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor=PALETTE["paper"])
    style_axes(ax, "Label positive rate by bank-year",
               "Mean of E/S/G/commitment flags per (bank, year).")
    rates["cell"] = rates["bank"].astype(str) + " " + rates["year"].astype(str)
    x = np.arange(len(rates))
    for k, c in zip(["is_env", "is_soc", "is_gov", "is_commitment"],
                    [PALETTE["accent2"], PALETTE["highlight"], PALETTE["accent"], PALETTE["ink"]]):
        ax.plot(x, rates[k], marker="o", label=k.replace("is_", ""), color=c)
    ax.set_xticks(x, rates["cell"], rotation=60, ha="right", fontsize=7)
    ax.legend(frameon=False)
    ax.set_ylabel("Positive rate")
    style_axes(ax2, "Specificity-level mix",
               f"Entropy = {spec['entropy_bits']:.2f} bits, "
               f"effective states = {spec['effective_states']:.2f}.")
    lv = spec["counts"]
    colmap = {0: SCORE_CMAP(0.1), 1: SCORE_CMAP(0.55), 2: SCORE_CMAP(0.95)}
    ax2.bar([str(k) for k in lv], list(lv.values()),
            color=[colmap[k] for k in lv],
            edgecolor=PALETTE["paper"])
    ax2.set_xlabel("spec_level (0 vague / 1 named / 2 quantified)")
    ax2.set_ylabel("Commitment chunks")
    return _save(fig, out_dir, "corpus_labels.png")


def fig_noise_retention(clf, out_dir: Path) -> Path:
    """Chunks per bank-year as a proxy for retained prose volume after noise filtering."""
    by = clf.groupby(["bank", "year"]).size().rename("chunks").reset_index()
    by["cell"] = by["bank"].astype(str) + " " + by["year"].astype(str)
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=PALETTE["paper"])
    style_axes(ax, "Retained chunks per report",
               "Prose chunks kept after table/boilerplate noise filtering.")
    ax.bar(by["cell"], by["chunks"], color=PALETTE["accent2"], edgecolor=PALETTE["paper"])
    ax.set_xticks(range(len(by["cell"])), by["cell"], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Chunks")
    return _save(fig, out_dir, "corpus_noise_retention.png")


def main(out_dir: str = "experiments/figures", clf=None) -> list:
    apply_rcparams()
    from esgwash.eda.anon import anonymize
    if clf is None:
        clf = ce.load_classified()
    clf = anonymize(clf)
    out = Path(out_dir)
    return [fig_token_lattice(clf, out), fig_coverage(clf, out),
            fig_labels(clf, out), fig_noise_retention(clf, out)]


if __name__ == "__main__":
    for p in main():
        print(f"-> {p}")
