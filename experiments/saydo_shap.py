"""RQ5 driver: chunk-level GBM (CV macro-F1) + SHAP on spec_level, plus
per-pillar say-do heatmap and rubric figure-type bar. Panel views are
EXPLORATORY (N=45); the model is chunk-level (large N)."""
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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from esgwash.eda.style import DELTA_CMAP, PALETTE, SCORE_CMAP, apply_rcparams, style_axes, symmetric_delta_norm
from esgwash.indices.figures_extract import figure_table, pillar_say_do
from esgwash.models.spec_features import FEATURE_COLS, chunk_features


def _load_classified() -> pd.DataFrame:
    files = sorted(glob.glob("outputs/cti/*/*/classified.parquet"))
    frames = [pd.read_parquet(f) for f in files]
    return pd.concat([f for f in frames if not f.empty], ignore_index=True)


def train_eval(X, y, seed: int = 42) -> dict:
    y = pd.Series(y).reset_index(drop=True)
    X = pd.DataFrame(X).reset_index(drop=True)
    k = int(min(3, y.value_counts().min()))
    model = GradientBoostingClassifier(random_state=seed)
    if k >= 2:
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        f1s = []
        for tr, te in skf.split(X, y):
            m = GradientBoostingClassifier(random_state=seed).fit(X.iloc[tr], y.iloc[tr])
            f1s.append(f1_score(y.iloc[te], m.predict(X.iloc[te]), average="macro"))
        cv = float(np.mean(f1s))
    else:
        cv = float("nan")
    model.fit(X, y)
    return {"macro_f1_cv": cv, "model": model, "n": int(len(X))}


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    fig.savefig(p, bbox_inches="tight", facecolor=PALETTE["paper"])
    plt.close(fig)
    return p


def fig_shap(model, X, out_dir: Path) -> Path:
    import shap
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(X)
    fig = plt.figure(facecolor=PALETTE["paper"])
    shap.summary_plot(sv, X, show=False, plot_type="bar", class_names=["vague", "named", "quantified"])
    return _save(fig, out_dir, "shap_spec_level.png")


def fig_say_do(classified, out_dir: Path) -> Path:
    sd = pillar_say_do(classified)
    grid = sd.pivot_table(index=["bank", "year"], columns="pillar", values="say_do")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(grid))), facecolor=PALETTE["paper"])
    im = ax.imshow(grid.to_numpy(), aspect="auto", cmap=DELTA_CMAP,
                   norm=symmetric_delta_norm(grid.to_numpy().ravel()))
    ax.set_title("Say-do gap per pillar (CTI_p - QDR_p), exploratory", loc="left",
                 fontweight="bold", color=PALETTE["ink"], pad=20)
    ax.set_xticks(range(len(grid.columns)), grid.columns)
    ax.set_yticks(range(len(grid.index)), [f"{b} {y}" for b, y in grid.index], fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="say - do")
    return _save(fig, out_dir, "index_say_do.png"), sd


def fig_figure_types(classified, out_dir: Path):
    ft = figure_table(classified)
    tot = ft.groupby("type")["n"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=PALETTE["paper"])
    style_axes(ax, "Quantified disclosures by type",
               "Rubric-extracted figures across all ESG-commitment chunks.")
    ax.bar(tot.index, tot.values, color=SCORE_CMAP(np.linspace(0.1, 0.95, len(tot))),
           edgecolor=PALETTE["paper"])
    ax.set_ylabel("Figure count")
    ax.set_xticks(range(len(tot.index)), tot.index, rotation=30, ha="right")
    return _save(fig, out_dir, "figure_types.png"), ft


def main(out_dir_fig: str = "experiments/figures", out_dir_tab: str = "experiments/panel",
         classified=None) -> dict:
    apply_rcparams()
    if classified is None:
        classified = _load_classified()
    fig_dir, tab_dir = Path(out_dir_fig), Path(out_dir_tab)
    tab_dir.mkdir(parents=True, exist_ok=True)

    X, y = chunk_features(classified)
    res = train_eval(X, y)
    fig_shap(res["model"], X, fig_dir)
    imp = pd.DataFrame({"feature": FEATURE_COLS,
                        "importance": res["model"].feature_importances_}
                       ).sort_values("importance", ascending=False)

    _, sd = fig_say_do(classified, fig_dir)
    sd.to_csv(tab_dir / "say_do.csv", index=False)
    _, ft = fig_figure_types(classified, fig_dir)
    ft.to_csv(tab_dir / "figure_types.csv", index=False)

    print(f"RQ5 spec_level GBM: n={res['n']}, macro-F1 (CV) = {res['macro_f1_cv']:.3f}")
    print("Top features:\n" + imp.head(6).to_string(index=False))
    return res


if __name__ == "__main__":
    main()
