"""3 figures báo cáo tuning + training (nhãn EN). Lưu figures/phase02/.

  1) tuning_trials_by_task.png — 10 Optuna trial/task (random vs TPE), best ★ (val macro-F1)
  2) phobert_vs_baseline.png   — test macro-F1: Majority / TF-IDF+LR / PhoBERT
  3) val_vs_test_gap.png       — val vs test macro-F1 (gap commit/spec)

Chạy: python outputs/_review_training_figures.py
"""
import json
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import optuna  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
optuna.logging.set_verbosity(optuna.logging.WARNING)
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

TASKS = ["env", "soc", "gov", "commitment", "specificity"]
FIG = Path("figures/phase02")
FIG.mkdir(parents=True, exist_ok=True)
C_MAJ, C_TFIDF, C_PHO, C_VAL = "#bdbdbd", "#34a853", "#1a73e8", "#f9ab00"


def fig_tuning() -> None:
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(10, 6))
    for xi, task in enumerate(TASKS):
        name = f"esg_{task}_hyperopt"
        study = optuna.load_study(study_name=name, storage=f"sqlite:///{name}.db")
        trials = sorted([t for t in study.trials if t.value is not None], key=lambda t: t.number)
        vals = [t.value for t in trials]
        xs = rng.normal(xi, 0.06, len(vals))
        for t, x, v in zip(trials, xs, vals):
            c = C_MAJ if t.number < 5 else C_PHO  # gray=random startup, blue=TPE
            ax.scatter(x, v, color=c, s=45, zorder=3, edgecolor="white", linewidth=0.5)
        best = max(vals)
        ax.scatter(xi, best, marker="*", s=320, color="#e8710a", zorder=4, edgecolor="black", linewidth=0.6)
        ax.text(xi, best + 0.004, f"{best:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(TASKS)))
    ax.set_xticklabels(TASKS)
    ax.set_ylabel("Validation macro-F1")
    ax.set_title("Hyperparameter tuning — 10 Optuna trials per task (optimized on validation)\n"
                 "gray = random startup (#0–4) · blue = TPE (#5–9) · ★ = best (used for final training)",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_MAJ, label="random startup", markersize=9),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_PHO, label="TPE", markersize=9),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#e8710a", label="best", markersize=15),
    ], loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "tuning_trials_by_task.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved tuning_trials_by_task.png")


def _load_summary() -> dict:
    return json.loads(Path("outputs/compare/phobert_test_summary.json").read_text(encoding="utf-8"))


def fig_baseline() -> None:
    s = _load_summary()
    maj = [s[t]["baseline_test"]["majority"] for t in TASKS]
    tf = [s[t]["baseline_test"]["tfidf_lr"] for t in TASKS]
    pb = [s[t]["phobert_test"]["macro_f1"] for t in TASKS]
    x = np.arange(len(TASKS))
    w = 0.27
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - w, maj, w, label="Majority", color=C_MAJ, edgecolor="black", linewidth=0.4)
    ax.bar(x, tf, w, label="TF-IDF + LogReg", color=C_TFIDF, edgecolor="black", linewidth=0.4)
    bars = ax.bar(x + w, pb, w, label="PhoBERT", color=C_PHO, edgecolor="black", linewidth=0.4)
    for xi, b, p, t in zip(x, bars, pb, tf):
        ax.text(b.get_x() + b.get_width() / 2, p + 0.012, f"{p:.2f}\n(+{p-t:.2f})",
                ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(TASKS)
    ax.set_ylabel("Test macro-F1")
    ax.set_ylim(0, 1.08)
    ax.set_title("Test macro-F1 — PhoBERT vs baselines (vi_gold, held-out test)\n"
                 "PhoBERT beats TF-IDF+LR on every task (+0.06…+0.09); crushes majority",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "phobert_vs_baseline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved phobert_vs_baseline.png")


def fig_gap() -> None:
    s = _load_summary()
    val = [s[t]["val_f1"] for t in TASKS]
    test = [s[t]["phobert_test"]["macro_f1"] for t in TASKS]
    x = np.arange(len(TASKS))
    w = 0.36
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - w / 2, val, w, label="Validation (optimistic)", color=C_VAL, edgecolor="black", linewidth=0.4)
    ax.bar(x + w / 2, test, w, label="Test (honest)", color=C_PHO, edgecolor="black", linewidth=0.4)
    for xi, v, te in zip(x, val, test):
        gap = te - v
        ax.text(xi, max(v, te) + 0.012, f"{gap:+.2f}", ha="center", fontsize=10,
                color="crimson" if gap < -0.05 else "gray", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(TASKS)
    ax.set_ylabel("macro-F1")
    ax.set_ylim(0, 1.08)
    ax.set_title("Validation vs Test macro-F1 — gap = optimism + distribution shift\n"
                 "commit/spec −0.12 = Bingler train/test split shift (report TEST, not val)",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "val_vs_test_gap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved val_vs_test_gap.png")


if __name__ == "__main__":
    fig_tuning()
    fig_baseline()
    fig_gap()
    print("Done -> figures/phase02/")
