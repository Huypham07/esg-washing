"""Figures demo P05 (nhãn EN theo convention figure).

  1) CTI ranking per bank by pillar (bar + bootstrap CI)
  2) Selective-disclosure heatmap (10 banks x E/S/G share)

Đọc enriched_corpus_<track>.parquet. Lưu figures/phase05/.
Chạy: python src/pipeline/cti_figures.py [--track silver]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.pipeline.classify_corpus import load_cti_config  # noqa: E402

PILLARS = ["E", "S", "G"]
PNAME = {"E": "Environmental", "S": "Social", "G": "Governance"}
PCOLOR = {"E": "#2e8b57", "S": "#4169e1", "G": "#cd5c5c"}


def bank_pillar_cti(enriched: pd.DataFrame, B: int = 1000, seed: int = 42) -> pd.DataFrame:
    """CTI gộp theo (bank,pillar) qua mọi năm + bootstrap CI 95%."""
    rng = np.random.default_rng(seed)
    rows = []
    for pillar in PILLARS:
        sub = enriched[enriched[f"is_{pillar}"] == 1]
        for bank, g in sub.groupby("bank"):
            commits = g[g.commitment == 1]
            n = len(commits)
            if n == 0:
                rows.append(dict(bank=bank, pillar=pillar, cti=np.nan, lo=np.nan, hi=np.nan, n=0))
                continue
            vague = (commits.specificity == 0).to_numpy().astype(float)
            cti = float(vague.mean())
            boot = rng.choice(vague, size=(B, n), replace=True).mean(axis=1)
            lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
            rows.append(dict(bank=bank, pillar=pillar, cti=cti, lo=lo, hi=hi, n=int(n)))
    return pd.DataFrame(rows)


def bank_overall_cti(enriched: pd.DataFrame, B: int = 1000, seed: int = 42) -> pd.DataFrame:
    """CTI GỘP per bank — pool MỌI cam kết ESG (KHÔNG chia pillar) + bootstrap CI."""
    rng = np.random.default_rng(seed)
    esg = enriched[enriched.is_esg == 1]
    rows = []
    for bank, g in esg.groupby("bank"):
        commits = g[g.commitment == 1]
        n = len(commits)
        if n == 0:
            rows.append(dict(bank=bank, cti=np.nan, lo=np.nan, hi=np.nan, n=0))
            continue
        vague = (commits.specificity == 0).to_numpy().astype(float)
        cti = float(vague.mean())
        boot = rng.choice(vague, size=(B, n), replace=True).mean(axis=1)
        lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
        rows.append(dict(bank=bank, cti=cti, lo=lo, hi=hi, n=int(n)))
    return pd.DataFrame(rows)


def fig_cti_ranking_overall(bo: pd.DataFrame, out_path: Path, track: str) -> None:
    """1 chart gộp: CTI tổng/bank (mọi pillar pool chung), xếp hạng + CI."""
    d = bo.dropna(subset=["cti"]).sort_values("cti").reset_index(drop=True)
    y = np.arange(len(d))
    err = np.vstack([d.cti - d.lo, d.hi - d.cti])
    colors = plt.cm.RdYlGn_r(plt.Normalize(0.3, 0.8)(d.cti.to_numpy()))  # xanh=thấp, đỏ=cao
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(y, d.cti, xerr=err, color=colors, capsize=3, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(d.bank)
    ax.set_xlim(0, 1.0)
    ax.axvline(0.5, ls="--", color="gray", lw=0.8)
    m = float(d.cti.mean())
    ax.axvline(m, ls=":", color="black", lw=1.2, label=f"mean = {m:.2f}")
    for yi, (_, r) in zip(y, d.iterrows()):
        ax.text(min(r.hi + 0.02, 0.97), yi, f"{r.cti:.2f} (n={r.n})", va="center", fontsize=8)
    ax.set_xlabel("CTI = share of vague commitments (all ESG pillars pooled)")
    ax.set_title(f"Overall CTI per bank — {track} (DEMO, pending Phase 04)\n"
                 f"all ESG commitments pooled  ·  higher = vaguer = more washing",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def fig_cti_ranking(bp: pd.DataFrame, out_path: Path, track: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True)
    for ax, pillar in zip(axes, PILLARS):
        d = bp[(bp.pillar == pillar) & bp.cti.notna()].sort_values("cti")
        y = np.arange(len(d))
        err = np.vstack([d.cti - d.lo, d.hi - d.cti])
        ax.barh(y, d.cti, xerr=err, color=PCOLOR[pillar], capsize=3,
                alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(d.bank, fontsize=9)
        ax.set_title(PNAME[pillar], fontsize=12, fontweight="bold")
        ax.set_xlim(0, 1.12)
        ax.axvline(0.5, ls="--", color="gray", lw=0.8)
        ax.set_xlabel("CTI (share of vague commitments)")
        for yi, (_, r) in zip(y, d.iterrows()):
            ax.text(min(r.hi + 0.02, 1.05), yi, f"n={r.n}", va="center", fontsize=7, color="gray")
    fig.suptitle(f"CTI per bank by pillar — {track} (DEMO, pending Phase 04)   ·   "
                 f"higher = vaguer commitments = more washing", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def fig_selective_heatmap(enriched: pd.DataFrame, out_path: Path, track: str) -> None:
    esg = enriched[enriched.is_esg == 1]
    banks = sorted(esg.bank.unique())
    share = np.zeros((len(banks), 3))
    counts = np.zeros((len(banks), 3), dtype=int)
    for i, b in enumerate(banks):
        g = esg[esg.bank == b]
        c = [int((g[f"is_{p}"] == 1).sum()) for p in PILLARS]
        tot = sum(c)
        counts[i] = c
        share[i] = [x / tot if tot else 0.0 for x in c]

    fig, ax = plt.subplots(figsize=(6.5, 7))
    im = ax.imshow(share, cmap="YlGnBu", aspect="auto", vmin=0, vmax=max(0.6, float(share.max())))
    ax.set_xticks(range(3))
    ax.set_xticklabels([PNAME[p] for p in PILLARS])
    ax.set_yticks(range(len(banks)))
    ax.set_yticklabels(banks)
    for i in range(len(banks)):
        for j in range(3):
            mark = " *" if share[i, j] < 0.10 else ""
            ax.text(j, i, f"{share[i, j]*100:.0f}%{mark}\n({counts[i, j]})", ha="center", va="center",
                    fontsize=8, color="white" if share[i, j] > 0.45 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Share of bank's ESG sentences in pillar")
    ax.set_title(f"Selective disclosure — {track} (DEMO)\n"
                 f"share of ESG sentences per pillar   ·   * = <10% (under-disclosed)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main(args=None) -> None:
    p = argparse.ArgumentParser(description="Figures demo P05")
    p.add_argument("--config", default="config/cti.yml")
    p.add_argument("--track", default=None)
    a = p.parse_args(args)

    config = load_cti_config(a.config)
    track = a.track or config["track"]
    enriched = pd.read_parquet(Path(config["paths"]["out_dir"]) / f"enriched_corpus_{track}.parquet")

    fig_dir = Path("figures/phase05")
    fig_dir.mkdir(parents=True, exist_ok=True)
    bp = bank_pillar_cti(enriched, B=config.get("cti", {}).get("bootstrap_B", 1000),
                         seed=config.get("seed", 42))
    bp.to_csv(fig_dir / f"bank_pillar_cti_{track}.csv", index=False)
    fig_cti_ranking(bp, fig_dir / f"cti_ranking_by_pillar_{track}.png", track)

    bo = bank_overall_cti(enriched, B=config.get("cti", {}).get("bootstrap_B", 1000), seed=config.get("seed", 42))
    bo.to_csv(fig_dir / f"bank_overall_cti_{track}.csv", index=False)
    fig_cti_ranking_overall(bo, fig_dir / f"cti_ranking_overall_{track}.png", track)

    fig_selective_heatmap(enriched, fig_dir / f"selective_disclosure_heatmap_{track}.png", track)
    print("Done -> figures/phase05/")


if __name__ == "__main__":
    main()
