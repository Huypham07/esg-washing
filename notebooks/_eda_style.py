"""Shared publication-style helpers for Phase 01 EDA notebooks.

Import these — do NOT copy plotting boilerplate into each notebook (DRY).
Figures export to figures/phase01/ as PNG (300 dpi) + PDF for the paper.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def project_root() -> Path:
    """Find repo root (dir containing both src/ and data/) from cwd upward."""
    p = Path.cwd()
    for cand in [p, *p.parents]:
        if (cand / "src").is_dir() and (cand / "data").is_dir():
            return cand
    return p


ROOT = project_root()
FIG_DIR = ROOT / "figures" / "phase01"


def setup() -> tuple[Path, Path]:
    """Set publication style, make src importable, ensure figures dir. Returns (ROOT, FIG_DIR)."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.25)
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "DejaVu Sans",  # has Vietnamese diacritics for data samples
    })
    return ROOT, FIG_DIR


def savefig(name: str) -> None:
    """Save the current figure as PNG (300 dpi) + PDF into figures/phase01/."""
    for ext in ("png", "pdf"):
        plt.savefig(FIG_DIR / f"{name}.{ext}")
    print(f"saved -> figures/phase01/{name}.png (+pdf)")
