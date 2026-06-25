"""
experiments/iaa_atomic.py
Reproducible atomic-flag IAA table + S/G contrast figure.

Usage:
    python experiments/iaa_atomic.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIELDS = [
    "co_cam_ket",
    "co_hanh_dong_ten",
    "co_so_dinh_luong",
    "quy_ve_bank",
    "co_moc_tg",
    "g_soc",
    "g_gov",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
EVAL_DIR = REPO_ROOT / "experiments" / "eval"


# ---------------------------------------------------------------------------
# Pure function
# ---------------------------------------------------------------------------

def atomic_iaa(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Compute per-field IAA (Cohen's κ) between two annotator DataFrames.

    Parameters
    ----------
    a, b : DataFrames that both contain ``chunk_id`` plus any subset of FIELDS.

    Returns
    -------
    DataFrame with columns [field, n, agree, kappa, n_disagree], one row per
    field that appears in both a and b.  Fields absent from either frame are
    silently skipped.
    """
    merged = a.merge(b, on="chunk_id", suffixes=("_a", "_b"))

    rows = []
    for field in FIELDS:
        col_a = f"{field}_a"
        col_b = f"{field}_b"

        # Handle the case where the field wasn't suffixed (only one df has it)
        if field in merged.columns and col_a not in merged.columns:
            # field existed in only one — skip
            continue
        if col_a not in merged.columns or col_b not in merged.columns:
            continue

        sub = merged[[col_a, col_b]].dropna()
        n = len(sub)
        if n == 0:
            continue

        va = sub[col_a].values
        vb = sub[col_b].values

        agree = int((va == vb).sum())
        n_disagree = n - agree

        # kappa = nan if either side is constant
        if len(np.unique(va)) == 1 or len(np.unique(vb)) == 1:
            kappa = float("nan")
        else:
            kappa = float(cohen_kappa_score(va, vb))

        rows.append({
            "field": field,
            "n": n,
            "agree": agree,
            "kappa": kappa,
            "n_disagree": n_disagree,
        })

    return pd.DataFrame(rows, columns=["field", "n", "agree", "kappa", "n_disagree"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    annot1_path = DATA_DIR / "gold_annot_1_relabeled.xlsx"
    annot2_path = DATA_DIR / "gold_annot_2_relabeled.xlsx"

    if not annot1_path.exists():
        sys.exit(f"ERROR: {annot1_path} not found")
    if not annot2_path.exists():
        sys.exit(f"ERROR: {annot2_path} not found")

    a = pd.read_excel(annot1_path, sheet_name="Sheet1")
    b = pd.read_excel(annot2_path, sheet_name="Sheet1")

    result = atomic_iaa(a, b)

    print("\n=== Atomic IAA (Cohen kappa) ===")
    print(result.to_string(index=False))
    print()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # --- JSON ---
    json_path = EVAL_DIR / "iaa_atomic.json"
    records = result.copy()
    records["kappa"] = records["kappa"].apply(
        lambda x: None if (isinstance(x, float) and math.isnan(x)) else x
    )
    records.to_json(json_path, orient="records", indent=2)
    print(f"Saved: {json_path}")

    # --- Figure ---
    _save_figure(result)


def _save_figure(result: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))

    fields = result["field"].tolist()
    kappas = result["kappa"].tolist()

    # Colour: S/G fields highlighted differently
    sg_fields = {"g_soc", "g_gov"}
    colors = ["#2196F3" if f not in sg_fields else "#FF9800" for f in fields]

    bars = ax.bar(range(len(fields)), kappas, color=colors, edgecolor="white", width=0.6)

    ax.set_xticks(range(len(fields)))
    ax.set_xticklabels(fields, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Cohen's κ", fontsize=10)
    ax.set_title("Atomic-Flag IAA: Cohen κ per Field\n(orange = S/G topic flags — lower reliability)", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.8, color="gray", linestyle="--", linewidth=0.8, label="κ = 0.80 threshold")

    # Value labels
    for bar, kval in zip(bars, kappas):
        if not math.isnan(kval):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                kval + 0.01,
                f"{kval:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # Legend: colour meaning
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2196F3", label="Atomic commitment flags"),
        Patch(facecolor="#FF9800", label="Topic flags (S/G — lower IAA)"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right")

    fig.tight_layout()
    png_path = EVAL_DIR / "iaa_atomic.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
