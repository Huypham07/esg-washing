"""Bộ biểu đồ trình bày kết quả P3-L1 cho đồng nghiệp (đọc BẢN CUỐI cti_p3l1, NO GPU).

Dùng điểm production duy nhất (floor 0.35) thay band. 6 biểu đồ:
(1) lỗ hổng Mức 1 → (2) độ phủ grounding → (3) dải CTI + gCTI-L1 → (4) phân bố support →
(5) % đạt ngưỡng → (6) gCTI-L1 riêng theo trụ.
-> experiments/analyse_bidv/presentation/*.png
  python experiments/figs_l1_presentation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd              # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from esgwash.run import to_long   # noqa: E402

PILLARS = ["env", "soc", "gov"]
PUP = [p.upper() for p in PILLARS]
YEARS = [2023, 2024]
BANK = "bidv"
B0 = "outputs/cti"               # baseline (đồng nghiệp)
PROD = "outputs/cti_p3l1"        # BẢN CUỐI P3-L1 (floor 0.35, production)
THETA_L1 = 0.4
OUT = Path("experiments/analyse_bidv/presentation")


def load(root: str, year: int):
    d = Path(root) / BANK / str(year)
    clf = pd.read_parquet(d / "classified.parquet")
    gp = d / "claims_grounded.parquet"
    return clf, (pd.read_parquet(gp) if gp.exists() else None)


def gl1_pillar(root: str, yr: int) -> dict:
    """gCTI-L1 (refine Mức 1) per trụ: cheap = Mức0 | (Mức1 & support<θ_L1). Mức 2 giữ substantive."""
    clf, g = load(root, yr)
    cm = clf[clf["is_commitment"] == 1].copy()
    cm["_s"] = cm["chunk_index"].map(g.set_index("chunk_index")["support"]).fillna(0.0)
    long = to_long(cm)
    out = {}
    for p in PILLARS:
        sub = long[long["pillar"] == p]
        if not len(sub):
            out[p] = None
            continue
        sl, s = sub["spec_level"].to_numpy(), sub["_s"].to_numpy()
        out[p] = round(float(((sl == 0) | ((sl == 1) & (s < THETA_L1))).mean()), 3)
    return out


def _m1_esg(clf: pd.DataFrame) -> pd.DataFrame:
    """Chunk commitment Mức 1 & thuộc ≥1 trụ E/S/G (khớp mẫu số gCTI per-trụ; loại non-ESG)."""
    return clf[(clf.is_commitment == 1) & (clf.spec_level == 1)
               & (clf[[f"is_{p}" for p in PILLARS]].sum(axis=1) > 0)]


def pct_m1(root: str, yr: int, ge: float) -> float:
    """% hành động Mức 1 (ESG) có support >= ge (ge=0 -> support>0)."""
    clf, g = load(root, yr)
    m1 = _m1_esg(clf)
    s = m1["chunk_index"].map(g.set_index("chunk_index")["support"]).fillna(0.0)
    cond = (s > 0) if ge == 0 else (s >= ge)
    return round(float(cond.mean() * 100), 1)


def fig1_phanbo():
    fig, axes = plt.subplots(1, len(YEARS), figsize=(11, 4.2))
    for ax, yr in zip(axes, YEARS):
        clf, _ = load(B0, yr)
        long = to_long(clf[clf["is_commitment"] == 1])
        b0 = [int(((long.pillar == p) & (long.spec_level == 0)).sum()) for p in PILLARS]
        b1 = [int(((long.pillar == p) & (long.spec_level == 1)).sum()) for p in PILLARS]
        b2 = [int(((long.pillar == p) & (long.spec_level == 2)).sum()) for p in PILLARS]
        x = np.arange(len(PILLARS))
        ax.bar(x, b0, label="Mức 0 — mơ hồ", color="tab:red")
        ax.bar(x, b1, bottom=b0, label="Mức 1 — hành động (lỗ hổng)", color="gold")
        ax.bar(x, b2, bottom=np.array(b0) + np.array(b1), label="Mức 2 — định lượng", color="tab:green")
        ax.set_xticks(x); ax.set_xticklabels(PUP); ax.set_ylabel("số cam kết")
        ax.set_title(f"{BANK.upper()} {yr}"); ax.legend(fontsize=8)
    fig.suptitle("(1) Lỗ hổng: Mức 1 (~1/3 cam kết) — baseline luôn coi là cheap talk", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "01_phan_bo_3_muc.png", dpi=130); plt.close(fig)


def fig2_dophu():
    fig, ax = plt.subplots(figsize=(7, 4.3))
    after = [pct_m1(PROD, yr, 0) for yr in YEARS]
    x = np.arange(len(YEARS)); w = 0.35
    ax.bar(x - w / 2, [0] * len(YEARS), w, label="Baseline P3 (đồng nghiệp)", color="lightgray")
    bars = ax.bar(x + w / 2, after, w, label="P3-L1 (bản cuối)", color="tab:blue")
    for b, v in zip(bars, after):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([str(y) for y in YEARS])
    ax.set_ylabel("% hành động Mức 1 tìm được bằng chứng"); ax.set_ylim(0, 100)
    ax.set_title("(2) Độ phủ grounding Mức 1: 0% → có bằng chứng (support>0)"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "02_do_phu_grounding.png", dpi=130); plt.close(fig)


def fig3_band():
    fig, axes = plt.subplots(1, len(YEARS), figsize=(11, 4.4))
    for ax, yr in zip(axes, YEARS):
        clf, _ = load(B0, yr)
        long = to_long(clf[clf["is_commitment"] == 1])
        loose, strict = [], []
        for p in PILLARS:
            sub = long[long.pillar == p]
            loose.append(round(float((sub.spec_level == 0).mean()), 3) if len(sub) else 0)
            strict.append(round(float((sub.spec_level <= 1).mean()), 3) if len(sub) else 0)
        g = gl1_pillar(PROD, yr)
        x = np.arange(len(PILLARS))
        ax.bar(x - 0.2, loose, 0.4, label="CTI_loose (coi mọi Mức 1 = thực chất)", color="tab:green")
        ax.bar(x + 0.2, strict, 0.4, label="CTI_strict (coi mọi Mức 1 = cheap)", color="tab:red", alpha=.8)
        for i, p in enumerate(PILLARS):
            if g[p] is None:
                continue
            ax.plot(i, g[p], "D", color="black", markersize=9, zorder=5,
                    label="gCTI-L1 (data-driven, bản cuối)" if i == 0 else None)
            ax.text(i, g[p] + 0.03, f"{g[p]:.2f}", ha="center", fontsize=8, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(PUP); ax.set_ylim(0, 1)
        ax.set_ylabel("tỉ lệ cheap talk"); ax.set_title(f"{BANK.upper()} {yr}"); ax.legend(fontsize=7)
    fig.suptitle("(3) P3-L1 thu hẹp dải mơ hồ Mức 1 thành điểm data-driven (nằm giữa loose/strict)",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "03_cti_band_gcti_l1.png", dpi=130); plt.close(fig)


def fig4_support():
    fig, axes = plt.subplots(1, len(YEARS), figsize=(11, 4))
    for ax, yr in zip(axes, YEARS):
        clf, g = load(PROD, yr)
        m1 = _m1_esg(clf)
        s = m1["chunk_index"].map(g.set_index("chunk_index")["support"]).fillna(0.0)
        ax.hist(s, bins=20, range=(0, 1), color="tab:blue", alpha=.8)
        ax.axvline(THETA_L1, ls="--", color="red", lw=2, label=f"θ_L1 = {THETA_L1}")
        ax.set_xlabel("support (độ mạnh bằng chứng)"); ax.set_ylabel("số hành động Mức 1")
        ax.set_title(f"{BANK.upper()} {yr}"); ax.legend()
    fig.suptitle("(4) Phân bố điểm support của hành động Mức 1 (≥θ_L1 = được công nhận thực chất)",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "04_phan_bo_support.png", dpi=130); plt.close(fig)


def fig5_datnguong():
    fig, ax = plt.subplots(figsize=(7, 4.3))
    after = [pct_m1(PROD, yr, THETA_L1) for yr in YEARS]
    x = np.arange(len(YEARS)); w = 0.35
    ax.bar(x - w / 2, [0] * len(YEARS), w, label="Baseline P3 (đồng nghiệp)", color="lightgray")
    bars = ax.bar(x + w / 2, after, w, label="P3-L1 (bản cuối)", color="tab:blue")
    for b, v in zip(bars, after):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([str(y) for y in YEARS])
    ax.set_ylabel("% hành động Mức 1 đạt ngưỡng (support ≥ θ_L1=0.4)"); ax.set_ylim(0, 100)
    ax.set_title("(2b) Mức 1 ĐẠT NGƯỠNG = được công nhận thực chất"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "05_dat_nguong.png", dpi=130); plt.close(fig)


def fig6_gcti_l1_rieng():
    fig, axes = plt.subplots(1, len(YEARS), figsize=(11, 4.4), sharey=True)
    for ax, yr in zip(axes, YEARS):
        g = gl1_pillar(PROD, yr)
        vals = [g[p] if g[p] is not None else 0 for p in PILLARS]
        x = np.arange(len(PILLARS))
        ax.bar(x, vals, 0.55, color="tab:purple", alpha=.85)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(PUP); ax.set_ylim(0, 1)
        ax.set_title(f"BIDV {yr}"); ax.set_ylabel("tỉ lệ nói suông (gCTI-L1)")
    fig.suptitle("Tỉ lệ nói suông theo trụ — đo bằng grounding P3-L1 (càng cao càng nhiều cheap talk)",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "06_gcti_l1_rieng.png", dpi=130); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig1_phanbo(); fig2_dophu(); fig3_band(); fig4_support(); fig5_datnguong(); fig6_gcti_l1_rieng()
    print(f"-> {OUT}/ : 01..06 (.png) — nguồn: BẢN CUỐI {PROD} (production, l1_sim_floor theo grounding.yml)")


if __name__ == "__main__":
    main()
