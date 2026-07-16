# experiments/render_print_figures.py
"""Tái sinh figure bản in (P06) từ ĐÚNG artifact run B — chỉ đổi style, không đổi data.

Nguồn data (run B, bất biến):
- experiments/panel/panel.csv            -> ribbons VDR/NAR/QDR (cột cti = VDR, tên lịch sử)
- outputs/cti/*/*/pillar_shares.parquet  -> share theo trụ cột
- experiments/panel/figure_types.csv     -> đếm loại số liệu định lượng
- experiments/panel/say_do.csv           -> VDR_p (cti_p) vs QDR_p + gap theo trụ

Mọi số hiển thị được ASSERT khớp số FREEZE trong paper trước khi vẽ.
Kích thước hình = kích thước in thật (inch) để font in ra >= 6pt (floor Springer).
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from esgwash.eda import index_eda as ie
from esgwash.eda.style import PALETTE

DPI = 600
INK, MUTED, GRID = PALETTE["ink"], PALETTE["muted"], PALETTE["grid"]
ACCENT, ACCENT2, HILITE = PALETTE["accent"], PALETTE["accent2"], PALETTE["highlight"]
PAPER, PANEL = PALETTE["paper"], PALETTE["panel"]


def _base_style(ax, fs=8):
    ax.set_facecolor(PANEL)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=fs, width=0.6, length=2)
    ax.grid(axis="y", color=GRID, linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)


def fig_ribbons_print(out: Path) -> Path:
    panel = pd.read_csv("experiments/panel/panel.csv")
    means = panel[["cti", "nar", "qdr"]].mean().round(3)
    assert abs(means["cti"] - 0.479) < 0.0015, f"VDR mean lech freeze: {means['cti']}"
    assert abs(means["nar"] - 0.304) < 0.0015, f"NAR mean lech freeze: {means['nar']}"
    assert abs(means["qdr"] - 0.217) < 0.0015, f"QDR mean lech freeze: {means['qdr']}"

    # 1.00in (dẹt): nhường chỗ cho bảng per-bank chuyển vị; font pt giữ nguyên (scale hiển thị ~1.0)
    fig, axes = plt.subplots(1, 3, figsize=(4.8, 1.00), facecolor=PAPER, sharey=True)
    for ax, col, title in zip(axes, ["cti", "nar", "qdr"], ["VDR", "NAR", "QDR"]):
        rb = ie.quantile_ribbons(panel, col)
        yrs = rb.index.to_numpy()
        _base_style(ax)
        ax.fill_between(yrs, rb["q10"], rb["q90"], color=HILITE, alpha=0.18, linewidth=0)
        ax.fill_between(yrs, rb["q25"], rb["q75"], color=HILITE, alpha=0.35, linewidth=0)
        ax.plot(yrs, rb["mean"], color=ACCENT, linewidth=1.3, marker="o", markersize=2.2)
        ax.set_title(title, loc="left", fontsize=8, fontweight="bold", color=INK, pad=2)
        ax.set_xticks(yrs)
        ax.set_xticklabels([str(int(y)) for y in yrs], fontsize=6)
        ax.set_ylim(0, 0.72)
        ax.set_yticks([0, 0.2, 0.4, 0.6])
    fig.subplots_adjust(wspace=0.12)
    p = out / "index_ribbons_print.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return p


def fig3a_share_print(out: Path) -> Path:
    frames = []
    for f in sorted(glob.glob("outputs/cti/*/*/pillar_shares.parquet")):
        frames.append(pd.read_parquet(f))
    data = pd.concat(frames, ignore_index=True)
    m = data.groupby("pillar")["share"].mean()
    vals = [m["env"], m["soc"], m["gov"]]
    assert [round(v, 3) for v in vals] == [0.240, 0.357, 0.403], f"share lech freeze: {vals}"

    fig, ax = plt.subplots(figsize=(1.62, 1.42), facecolor=PAPER)
    _base_style(ax)
    colors = [ACCENT2, HILITE, ACCENT]
    bars = ax.bar(["E", "S", "G"], vals, color=colors, width=0.6, edgecolor=PAPER)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold", color=INK)
    ax.set_ylim(0, 0.48)
    ax.set_yticks([0, 0.2, 0.4])
    ax.set_ylabel("Mean share", fontsize=8, color=MUTED)
    ax.tick_params(axis="x", labelsize=8.5)
    p = out / "fig3a_share_print.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return p


def fig3b_figtypes_print(out: Path) -> Path:
    ft = pd.read_csv("experiments/panel/figure_types.csv")
    tot = ft.groupby("type")["n"].sum().sort_values(ascending=False)
    show = tot[[t for t in tot.index if t != "other"]]
    assert int(show["green_credit"]) == 90, f"green_credit lech freeze: {show['green_credit']}"
    assert int(show["emissions"]) == 9, f"emissions lech freeze: {show['emissions']}"

    fig, ax = plt.subplots(figsize=(1.62, 1.42), facecolor=PAPER)
    _base_style(ax, fs=8.5)
    ax.grid(axis="x", color=GRID, linewidth=0.4, alpha=0.6)
    ax.grid(axis="y", visible=False)
    labels = [t.replace("_", " ") for t in show.index][::-1]
    vals = show.values[::-1]
    cmap = plt.get_cmap("copper_r")
    ax.barh(labels, vals, color=[cmap(0.25 + 0.6 * v / max(vals)) for v in vals],
            height=0.65, edgecolor=PAPER)
    for i, v in enumerate(vals):
        ax.text(v + 1.5, i, str(int(v)), va="center", fontsize=8.5, color=MUTED)
    ax.set_xlim(0, 103)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.set_xticks([0, 45, 90])
    ax.set_xlabel("Count", fontsize=8.5, color=MUTED)
    p = out / "fig3b_figtypes_print.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return p


def fig3c_gap_print(out: Path) -> Path:
    sd = pd.read_csv("experiments/panel/say_do.csv")
    g = sd.groupby("pillar")[["cti_p", "qdr_p"]].mean()
    gaps = (g["cti_p"] - g["qdr_p"])
    order = ["env", "soc", "gov"]
    assert [round(gaps[p], 2) for p in order] == [0.19, 0.23, 0.41], f"gap lech freeze: {gaps.to_dict()}"

    fig, ax = plt.subplots(figsize=(1.72, 1.42), facecolor=PAPER)
    _base_style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.4, alpha=0.6)
    ax.grid(axis="y", visible=False)
    names = {"env": "E", "soc": "S", "gov": "G"}
    # QDR = vong tron RONG (khac shape/fill voi VDR dac) -> phan biet duoc khi in trang-den;
    # bo legend (truoc bi marker de len chu) -> nhan truc tiep tren hang E + gap ghi ben phai thanh
    for i, pill in enumerate(reversed(order)):
        y = i
        v, q = g.loc[pill, "cti_p"], g.loc[pill, "qdr_p"]
        ax.plot([q, v], [y, y], color=GRID, linewidth=2.2, zorder=1, solid_capstyle="round")
        ax.scatter([v], [y], s=26, color=ACCENT, zorder=3)
        ax.scatter([q], [y], s=30, facecolors=PAPER, edgecolors=ACCENT2,
                   linewidths=1.3, zorder=3)
        ax.text(v + 0.035, y, f"+{(v - q):.2f}", ha="left", va="center",
                fontsize=8, fontweight="bold", color=INK)
    v_e, q_e = g.loc["env", "cti_p"], g.loc["env", "qdr_p"]
    ax.text(v_e, 2.5, "VDR", ha="center", fontsize=8, fontweight="bold", color=ACCENT)
    ax.text(q_e, 2.5, "QDR", ha="center", fontsize=8, fontweight="bold", color=ACCENT2)
    ax.set_yticks(range(3), [names[p] for p in reversed(order)], fontsize=8.5)
    ax.set_ylim(-0.5, 2.95)
    ax.set_xlim(0, 0.72)
    ax.set_xticks([0, 0.2, 0.4, 0.6])
    p = out / "fig3c_gap_print.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="experiments/figures_print")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for fn in (fig_ribbons_print, fig3a_share_print, fig3b_figtypes_print, fig3c_gap_print):
        print(f"-> {fn(out)}")
    print("OK: moi so da assert khop freeze.")


if __name__ == "__main__":
    main()
