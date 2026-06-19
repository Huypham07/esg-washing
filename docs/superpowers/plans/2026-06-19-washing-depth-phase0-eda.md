# Phase 0 — Style module + EDA layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a shared figure-style module and the corpus + index EDA figures (scores-EDA notebook style, English) over existing pipeline outputs, with pure-compute helpers unit-tested and renderers smoke-tested.

**Architecture:** Pure, reusable compute helpers live in `src/esgwash/eda/` (unit-tested on toy DataFrames). Matplotlib renderers live in `experiments/eda_corpus.py` and `experiments/eda_indices.py` (smoke-tested: run end-to-end, assert PNG files written). Renderers read existing parquet outputs (`data/chunks.parquet`, `outputs/cti/*/*/classified.parquet`, `outputs/cti/*/*/pillar_shares.parquet`, `experiments/panel/panel.csv`) and import the shared style. No changes to L0 pipeline.

**Tech Stack:** Python 3.13, pandas, numpy, matplotlib 3.10, pytest. (No new dependency in this phase.)

## Global Constraints

- Figure content (titles, subtitles, legends, axis labels) MUST be in English; font `DejaVu Sans`.
- Do NOT compile LaTeX anywhere.
- Do NOT fabricate formulas/weights — EDA uses only counts, quantiles, Shannon entropy (`-Σ p log2 p`), and standard PCA via `numpy.linalg.eigh`.
- Entry/renderer scripts MUST bootstrap `sys.path` by inserting the repo `src/` dir so `from esgwash...` works on Kaggle (`sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))`).
- Do NOT modify the `classified.parquet` / `pillar_shares.parquet` / `panel.csv` contracts.
- Figures saved as PNG, `dpi=140`, into `experiments/figures/`. Tables (JSON) into `experiments/eda/`.
- Tests are flat in `tests/`, run with `pytest`, import as `from esgwash...`.

## File Structure

- Create `src/esgwash/eda/__init__.py` — exports style + compute helpers.
- Create `src/esgwash/eda/style.py` — palette, colormaps, `style_axes`, `shannon_entropy_bits`, `effective_states`.
- Create `src/esgwash/eda/corpus_eda.py` — pure compute: load classified, token distribution, label rates, spec-level distribution, coverage matrix, noise retention.
- Create `src/esgwash/eda/index_eda.py` — pure compute: washing feature matrix, manual PCA, trajectory deltas, quantile ribbons.
- Create `experiments/eda_corpus.py` — renderers for figures 1–4.
- Create `experiments/eda_indices.py` — renderers for figures 5–9.
- Create `tests/test_eda_style.py`, `tests/test_eda_corpus.py`, `tests/test_eda_index.py`.

Known schemas (verified):
- `chunks.parquet`: `chunk_id, doc_id, bank, year, chunk_index, content_text, char_count, token_count`.
- `classified.parquet`: above + `p_env, is_env, p_soc, is_soc, p_gov, is_gov, pillar_top, p_commitment, is_commitment, p_specific, spec_level, is_specific, spec_parse_ok, spec_rubric, spec_raw`.
- `pillar_shares.parquet`: `bank, year, pillar, n, share, industry_share, share_dev`.
- `panel.csv`: `bank, year, cti, nar, qdr, n_commit` (+ pillar columns from analyse_panel if present).

---

### Task 1: Shared style module

**Files:**
- Create: `src/esgwash/eda/__init__.py`
- Create: `src/esgwash/eda/style.py`
- Test: `tests/test_eda_style.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `PALETTE: dict[str,str]` keys: `paper, panel, ink, muted, grid, accent, accent2, highlight`.
  - `SCORE_CMAP`, `DELTA_CMAP` (`matplotlib.colors.LinearSegmentedColormap`), `DELTA_NORM` (`TwoSlopeNorm`, vmin -0.9 / vcenter 0 / vmax 0.9).
  - `style_axes(ax, title: str | None = None, subtitle: str | None = None, title_pad: int = 28) -> Axes`.
  - `shannon_entropy_bits(probs) -> float` (ignores zeros; `-Σ p log2 p`).
  - `effective_states(probs) -> float` (`2 ** shannon_entropy_bits(probs)`).
  - `apply_rcparams() -> None` (sets figure.dpi=140, font.family=DejaVu Sans, sizes).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eda_style.py
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from esgwash.eda import style


def test_palette_has_required_keys():
    for k in ["paper", "panel", "ink", "muted", "grid", "accent", "accent2", "highlight"]:
        assert k in style.PALETTE and style.PALETTE[k].startswith("#")


def test_entropy_uniform_two_states_is_one_bit():
    assert abs(style.shannon_entropy_bits([0.5, 0.5]) - 1.0) < 1e-9
    assert abs(style.effective_states([0.5, 0.5]) - 2.0) < 1e-9


def test_entropy_ignores_zero_prob():
    # zero-prob states must not produce NaN
    h = style.shannon_entropy_bits([0.0, 1.0])
    assert h == 0.0


def test_entropy_accepts_unnormalised_counts():
    # counts [1,1,1,1] -> 2 bits
    assert abs(style.shannon_entropy_bits([1, 1, 1, 1]) - 2.0) < 1e-9


def test_style_axes_returns_axes_and_sets_title():
    fig, ax = plt.subplots()
    out = style.style_axes(ax, title="Hello", subtitle="sub")
    assert out is ax
    assert ax.get_title(loc="left") == "Hello"
    plt.close(fig)


def test_delta_norm_centered_at_zero():
    assert style.DELTA_NORM.vcenter == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_eda_style.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'esgwash.eda'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/esgwash/eda/__init__.py
from esgwash.eda.style import (  # noqa: F401
    PALETTE, SCORE_CMAP, DELTA_CMAP, DELTA_NORM,
    style_axes, shannon_entropy_bits, effective_states, apply_rcparams,
)
```

```python
# src/esgwash/eda/style.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_eda_style.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/esgwash/eda/__init__.py src/esgwash/eda/style.py tests/test_eda_style.py
git commit -m "feat(eda): shared figure style + entropy helpers"
```

---

### Task 2: Corpus EDA compute helpers

**Files:**
- Create: `src/esgwash/eda/corpus_eda.py`
- Test: `tests/test_eda_corpus.py`

**Interfaces:**
- Consumes: pandas DataFrames matching `classified.parquet` schema.
- Produces:
  - `load_classified(root: str = "outputs/cti") -> pd.DataFrame` — concat all `*/*/classified.parquet` (empty DF with no error if none).
  - `token_distribution(chunks: pd.DataFrame, col: str = "token_count") -> dict` — keys `p10,q1,median,q3,p90,max,mean` (floats).
  - `label_positive_rates(clf: pd.DataFrame) -> pd.DataFrame` — index (bank,year) reset, columns `bank, year, is_env, is_soc, is_gov, is_commitment` = mean of each flag per cell.
  - `spec_level_distribution(clf: pd.DataFrame) -> dict` — `counts` (dict level->count over commitment ESG rows), `entropy_bits`, `effective_states`. Uses only `is_commitment==1 & (is_env|is_soc|is_gov)`.
  - `coverage_matrix(clf: pd.DataFrame, value: str = "chunk") -> pd.DataFrame` — pivot bank×year; `value="chunk"` counts rows, `value="commitment"` counts `is_commitment==1`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eda_corpus.py
import pandas as pd
from esgwash.eda import corpus_eda as ce


def _toy_clf():
    return pd.DataFrame({
        "bank": ["a", "a", "a", "b"], "year": [2023, 2023, 2023, 2023],
        "token_count": [10, 20, 30, 40],
        "is_env": [1, 0, 1, 0], "is_soc": [0, 1, 0, 0], "is_gov": [0, 0, 0, 1],
        "is_commitment": [1, 1, 0, 1], "spec_level": [0, 2, 0, 1],
    })


def test_token_distribution_quantiles():
    d = ce.token_distribution(_toy_clf())
    assert d["median"] == 25.0 and d["max"] == 40.0


def test_label_positive_rates_per_cell():
    out = ce.label_positive_rates(_toy_clf())
    a = out[out["bank"] == "a"].iloc[0]
    assert abs(a["is_env"] - (2 / 3)) < 1e-9
    assert abs(a["is_commitment"] - (2 / 3)) < 1e-9


def test_spec_level_distribution_only_esg_commitment():
    # commitment ESG rows: row0 (env,commit,L0), row1 (soc,commit,L2), row3 (gov,commit,L1)
    d = ce.spec_level_distribution(_toy_clf())
    assert d["counts"] == {0: 1, 1: 1, 2: 1}
    assert abs(d["entropy_bits"] - 1.584962500721156) < 1e-9  # log2(3)


def test_coverage_matrix_counts():
    cov = ce.coverage_matrix(_toy_clf(), value="chunk")
    assert cov.loc["a", 2023] == 3
    comm = ce.coverage_matrix(_toy_clf(), value="commitment")
    assert comm.loc["a", 2023] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_eda_corpus.py -v`
Expected: FAIL with `ModuleNotFoundError` / `AttributeError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/esgwash/eda/corpus_eda.py
"""Pure compute for corpus-level EDA (no plotting)."""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

from esgwash.eda.style import effective_states, shannon_entropy_bits

PILLARS = ("env", "soc", "gov")
_ESG = [f"is_{p}" for p in PILLARS]


def load_classified(root: str = "outputs/cti") -> pd.DataFrame:
    files = sorted(glob.glob(str(Path(root) / "*/*/classified.parquet")))
    frames = [pd.read_parquet(f) for f in files]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def token_distribution(chunks: pd.DataFrame, col: str = "token_count") -> dict:
    s = chunks[col].astype(float)
    q = s.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
    return {"p10": float(q.loc[0.10]), "q1": float(q.loc[0.25]),
            "median": float(q.loc[0.50]), "q3": float(q.loc[0.75]),
            "p90": float(q.loc[0.90]), "max": float(s.max()), "mean": float(s.mean())}


def label_positive_rates(clf: pd.DataFrame) -> pd.DataFrame:
    cols = _ESG + ["is_commitment"]
    return clf.groupby(["bank", "year"])[cols].mean().reset_index()


def _esg_commitment(clf: pd.DataFrame) -> pd.DataFrame:
    esg = clf[_ESG].max(axis=1).astype(bool)
    return clf[(clf["is_commitment"] == 1) & esg]


def spec_level_distribution(clf: pd.DataFrame) -> dict:
    sub = _esg_commitment(clf)
    counts = {int(k): int(v) for k, v in sub["spec_level"].value_counts().sort_index().items()}
    vals = list(counts.values())
    return {"counts": counts, "entropy_bits": shannon_entropy_bits(vals),
            "effective_states": effective_states(vals)}


def coverage_matrix(clf: pd.DataFrame, value: str = "chunk") -> pd.DataFrame:
    if value == "commitment":
        g = clf[clf["is_commitment"] == 1]
    else:
        g = clf
    return g.pivot_table(index="bank", columns="year", values="chunk_id",
                         aggfunc="count", fill_value=0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_eda_corpus.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/esgwash/eda/corpus_eda.py tests/test_eda_corpus.py
git commit -m "feat(eda): corpus-level compute helpers"
```

---

### Task 3: Corpus EDA renderers

**Files:**
- Create: `experiments/eda_corpus.py`
- Test: `tests/test_eda_corpus.py` (append smoke test)

**Interfaces:**
- Consumes: `esgwash.eda.style`, `esgwash.eda.corpus_eda`.
- Produces: `main(out_dir="experiments/figures", clf=None) -> list[Path]` — writes and returns PNG paths: `corpus_token_lattice.png`, `corpus_coverage.png`, `corpus_labels.png`, `corpus_noise_retention.png`. Accepts an in-memory `clf` DataFrame for testing (falls back to `load_classified()`).

- [ ] **Step 1: Write the failing test** (append to `tests/test_eda_corpus.py`)

```python
def test_eda_corpus_main_writes_pngs(tmp_path):
    import importlib.util, sys
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "eda_corpus", Path(__file__).resolve().parents[1] / "experiments" / "eda_corpus.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eda_corpus"] = mod
    spec.loader.exec_module(mod)
    clf = _toy_clf().assign(chunk_id=["a0", "a1", "a2", "b0"],
                            char_count=[50, 100, 150, 200])
    paths = mod.main(out_dir=str(tmp_path), clf=clf)
    assert len(paths) == 4
    for p in paths:
        assert p.exists() and p.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_eda_corpus.py::test_eda_corpus_main_writes_pngs -v`
Expected: FAIL (`FileNotFoundError` for experiments/eda_corpus.py).

- [ ] **Step 3: Write minimal implementation**

```python
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
from esgwash.eda.style import (PALETTE, SCORE_CMAP, apply_rcparams,
                               shannon_entropy_bits, style_axes)


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
    ax2.bar([str(k) for k in lv], list(lv.values()),
            color=[SCORE_CMAP(0.1), SCORE_CMAP(0.55), SCORE_CMAP(0.95)][:len(lv)],
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
    ax.set_xticklabels(by["cell"], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Chunks")
    return _save(fig, out_dir, "corpus_noise_retention.png")


def main(out_dir: str = "experiments/figures", clf=None) -> list:
    apply_rcparams()
    if clf is None:
        clf = ce.load_classified()
    out = Path(out_dir)
    return [fig_token_lattice(clf, out), fig_coverage(clf, out),
            fig_labels(clf, out), fig_noise_retention(clf, out)]


if __name__ == "__main__":
    for p in main():
        print(f"-> {p}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_eda_corpus.py -v`
Expected: PASS (5 passed total)

- [ ] **Step 5: Commit**

```bash
git add experiments/eda_corpus.py tests/test_eda_corpus.py
git commit -m "feat(eda): corpus EDA figures (token lattice, coverage, labels, retention)"
```

---

### Task 4: Index EDA compute helpers

**Files:**
- Create: `src/esgwash/eda/index_eda.py`
- Test: `tests/test_eda_index.py`

**Interfaces:**
- Consumes: `panel` DataFrame (`bank, year, cti, nar, qdr, n_commit`).
- Produces:
  - `washing_feature_matrix(panel: pd.DataFrame, cols=("cti","nar","qdr","n_commit")) -> tuple[np.ndarray, list[str], list[str]]` — returns (standardized matrix, row labels `"bank year"`, feature names). Standardize per column (z-score; zero-variance column -> zeros).
  - `manual_pca(X: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]` — returns (scores `[n_rows, n_components]`, loadings `[n_features, n_components]`, explained_variance_ratio `[n_components]`) using `numpy.linalg.eigh` on the covariance matrix, eigenvalues sorted descending.
  - `trajectory_deltas(panel: pd.DataFrame, col: str = "cti") -> dict[str, dict]` — per bank: `{"years": [...], "values": [...], "deltas": [...]}` (delta = year-over-year diff, first = 0.0), sorted by year.
  - `quantile_ribbons(panel: pd.DataFrame, col: str = "cti") -> pd.DataFrame` — index year, columns `q10,q25,median,q75,q90,mean`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eda_index.py
import numpy as np
import pandas as pd
from esgwash.eda import index_eda as ie


def _toy_panel():
    return pd.DataFrame({
        "bank": ["a", "a", "b", "b"], "year": [2022, 2023, 2022, 2023],
        "cti": [0.5, 0.4, 0.2, 0.3], "nar": [0.3, 0.3, 0.4, 0.4],
        "qdr": [0.2, 0.3, 0.4, 0.3], "n_commit": [10, 12, 8, 9],
    })


def test_feature_matrix_standardised():
    X, rows, feats = ie.washing_feature_matrix(_toy_panel())
    assert X.shape == (4, 4)
    assert rows[0] == "a 2022"
    # each column mean ~ 0 after z-score
    assert np.allclose(X.mean(axis=0), 0, atol=1e-9)


def test_manual_pca_shapes_and_variance():
    X, _, _ = ie.washing_feature_matrix(_toy_panel())
    scores, loadings, evr = ie.manual_pca(X, n_components=2)
    assert scores.shape == (4, 2)
    assert loadings.shape == (4, 2)
    assert evr[0] >= evr[1] and 0 <= evr[0] <= 1.0001


def test_trajectory_deltas_year_over_year():
    tr = ie.trajectory_deltas(_toy_panel(), "cti")
    assert tr["a"]["values"] == [0.5, 0.4]
    assert tr["a"]["deltas"][0] == 0.0
    assert abs(tr["a"]["deltas"][1] - (-0.1)) < 1e-9


def test_quantile_ribbons_per_year():
    rb = ie.quantile_ribbons(_toy_panel(), "cti")
    assert 2022 in rb.index and "median" in rb.columns
    assert abs(rb.loc[2022, "median"] - 0.35) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_eda_index.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/esgwash/eda/index_eda.py
"""Pure compute for index-level EDA (no plotting)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def washing_feature_matrix(panel: pd.DataFrame,
                           cols=("cti", "nar", "qdr", "n_commit")):
    cols = list(cols)
    rows = [f"{b} {y}" for b, y in zip(panel["bank"], panel["year"])]
    M = panel[cols].to_numpy(dtype=float)
    mu = M.mean(axis=0)
    sd = M.std(axis=0)
    sd[sd == 0] = 1.0
    X = (M - mu) / sd
    return X, rows, cols


def manual_pca(X: np.ndarray, n_components: int = 2):
    Xc = X - X.mean(axis=0)
    cov = np.cov(Xc, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)          # ascending
    order = np.argsort(vals)[::-1]            # descending
    vals, vecs = vals[order], vecs[:, order]
    loadings = vecs[:, :n_components]
    scores = Xc @ loadings
    total = vals.sum()
    evr = (vals[:n_components] / total) if total > 0 else np.zeros(n_components)
    return scores, loadings, evr


def trajectory_deltas(panel: pd.DataFrame, col: str = "cti") -> dict:
    out = {}
    for bank, g in panel.sort_values("year").groupby("bank"):
        vals = g[col].to_numpy(dtype=float).tolist()
        deltas = [0.0] + list(np.diff(vals))
        out[bank] = {"years": g["year"].tolist(), "values": vals, "deltas": deltas}
    return out


def quantile_ribbons(panel: pd.DataFrame, col: str = "cti") -> pd.DataFrame:
    g = panel.groupby("year")[col]
    rb = pd.DataFrame({
        "q10": g.quantile(0.10), "q25": g.quantile(0.25), "median": g.median(),
        "q75": g.quantile(0.75), "q90": g.quantile(0.90), "mean": g.mean()})
    return rb
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_eda_index.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/esgwash/eda/index_eda.py tests/test_eda_index.py
git commit -m "feat(eda): index-level compute (feature matrix, manual PCA, trajectory, ribbons)"
```

---

### Task 5: Index EDA renderers

**Files:**
- Create: `experiments/eda_indices.py`
- Test: `tests/test_eda_index.py` (append smoke test)

**Interfaces:**
- Consumes: `esgwash.eda.style`, `esgwash.eda.index_eda`; reads `experiments/panel/panel.csv` and `outputs/cti/*/*/pillar_shares.parquet`.
- Produces: `main(out_dir="experiments/figures", panel=None, shares=None) -> list[Path]` writing: `index_ribbons.png`, `index_cti_cartography.png`, `index_cti_trajectories.png`, `index_washing_pca.png`, `index_selective_disclosure.png`. Accepts in-memory `panel`/`shares` for testing.

- [ ] **Step 1: Write the failing test** (append to `tests/test_eda_index.py`)

```python
def test_eda_indices_main_writes_pngs(tmp_path):
    import importlib.util, sys
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "eda_indices", Path(__file__).resolve().parents[1] / "experiments" / "eda_indices.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eda_indices"] = mod
    spec.loader.exec_module(mod)
    panel = _toy_panel()
    shares = pd.DataFrame({
        "bank": ["a", "a", "a", "b", "b", "b"], "year": [2023] * 6,
        "pillar": ["env", "soc", "gov"] * 2,
        "share": [0.2, 0.5, 0.3, 0.4, 0.4, 0.2],
        "share_dev": [-0.1, 0.05, 0.05, 0.1, -0.05, -0.05]})
    paths = mod.main(out_dir=str(tmp_path), panel=panel, shares=shares)
    assert len(paths) == 5
    for p in paths:
        assert p.exists() and p.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_eda_index.py::test_eda_indices_main_writes_pngs -v`
Expected: FAIL (`FileNotFoundError` for experiments/eda_indices.py).

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/eda_indices.py
"""Index-level EDA figures (English, scores-EDA style)."""
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
from matplotlib.collections import LineCollection

from esgwash.eda import index_eda as ie
from esgwash.eda.style import (DELTA_CMAP, DELTA_NORM, PALETTE, SCORE_CMAP,
                               apply_rcparams, style_axes)


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    fig.savefig(p, bbox_inches="tight", facecolor=PALETTE["paper"])
    plt.close(fig)
    return p


def _load_panel() -> pd.DataFrame:
    return pd.read_csv("experiments/panel/panel.csv")


def _load_shares() -> pd.DataFrame:
    files = sorted(glob.glob("outputs/cti/*/*/pillar_shares.parquet"))
    frames = [pd.read_parquet(f) for f in files]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fig_ribbons(panel, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), facecolor=PALETTE["paper"], sharex=True)
    for ax, col in zip(axes, ["cti", "nar", "qdr"]):
        rb = ie.quantile_ribbons(panel, col)
        yrs = rb.index.to_numpy()
        style_axes(ax, col.upper(), "Yearly inter-quantile band.")
        ax.fill_between(yrs, rb["q10"], rb["q90"], color=PALETTE["highlight"], alpha=0.18)
        ax.fill_between(yrs, rb["q25"], rb["q75"], color=PALETTE["highlight"], alpha=0.35)
        ax.plot(yrs, rb["mean"], color=PALETTE["accent"], linewidth=2.4, marker="o")
        ax.set_xticks(yrs)
        ax.set_ylim(0, 1.02)
    return _save(fig, out_dir, "index_ribbons.png")


def fig_cti_cartography(panel, out_dir):
    grid = panel.pivot_table(index="bank", columns="year", values="cti")
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=PALETTE["paper"])
    im = ax.imshow(np.ma.masked_invalid(grid.to_numpy()), aspect="auto",
                   cmap=SCORE_CMAP, vmin=0, vmax=1)
    ax.set_title("CTI cartography (bank x year)", loc="left", fontweight="bold",
                 color=PALETTE["ink"], pad=20)
    ax.set_xticks(range(len(grid.columns)), grid.columns)
    ax.set_yticks(range(len(grid.index)), grid.index)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="CTI")
    return _save(fig, out_dir, "index_cti_cartography.png")


def fig_cti_trajectories(panel, out_dir):
    tr = ie.trajectory_deltas(panel, "cti")
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=PALETTE["paper"])
    style_axes(ax, "CTI trajectories",
               "Segment colour = year-over-year change (red rising / teal falling).")
    for bank, d in tr.items():
        yrs, vals, deltas = d["years"], d["values"], d["deltas"]
        pts = np.array([yrs, vals]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap=DELTA_CMAP, norm=DELTA_NORM)
        lc.set_array(np.array(deltas[1:]))
        lc.set_linewidth(2.6)
        ax.add_collection(lc)
        ax.text(yrs[-1], vals[-1], f" {bank}", fontsize=8, color=PALETTE["ink"], va="center")
    ax.set_xlim(min(min(d["years"]) for d in tr.values()) - 0.2,
               max(max(d["years"]) for d in tr.values()) + 0.8)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Year")
    ax.set_ylabel("CTI")
    return _save(fig, out_dir, "index_cti_trajectories.png")


def fig_washing_pca(panel, out_dir):
    X, rows, feats = ie.washing_feature_matrix(panel)
    scores, loadings, evr = ie.manual_pca(X, 2)
    fig, ax = plt.subplots(figsize=(9, 7), facecolor=PALETTE["paper"])
    style_axes(ax, "Washing-space PCA (bank-year)",
               f"PC1 {evr[0]:.0%} / PC2 {evr[1]:.0%} of variance.")
    ax.scatter(scores[:, 0], scores[:, 1], s=80, c=PALETTE["accent"],
               edgecolor=PALETTE["paper"], zorder=3)
    for (x, y), lab in zip(scores, rows):
        ax.text(x, y, f" {lab}", fontsize=7, color=PALETTE["ink"])
    for i, f in enumerate(feats):
        ax.annotate(f, xy=(loadings[i, 0] * 3, loadings[i, 1] * 3),
                    color=PALETTE["accent2"], fontsize=9,
                    arrowprops=dict(arrowstyle="->", color=PALETTE["accent2"]),
                    xytext=(0, 0), textcoords="offset points")
    ax.axhline(0, color=PALETTE["grid"], linewidth=0.8)
    ax.axvline(0, color=PALETTE["grid"], linewidth=0.8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    return _save(fig, out_dir, "index_washing_pca.png")


def fig_selective_disclosure(shares, out_dir):
    grid = shares.pivot_table(index=["bank", "year"], columns="pillar", values="share_dev")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(grid))), facecolor=PALETTE["paper"])
    im = ax.imshow(grid.to_numpy(), aspect="auto", cmap=DELTA_CMAP, norm=DELTA_NORM)
    ax.set_title("Selective disclosure (share deviation from industry)", loc="left",
                 fontweight="bold", color=PALETTE["ink"], pad=20)
    ax.set_xticks(range(len(grid.columns)), grid.columns)
    ax.set_yticks(range(len(grid.index)), [f"{b} {y}" for b, y in grid.index], fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="share - industry share")
    return _save(fig, out_dir, "index_selective_disclosure.png")


def main(out_dir: str = "experiments/figures", panel=None, shares=None) -> list:
    apply_rcparams()
    if panel is None:
        panel = _load_panel()
    if shares is None:
        shares = _load_shares()
    out = Path(out_dir)
    return [fig_ribbons(panel, out), fig_cti_cartography(panel, out),
            fig_cti_trajectories(panel, out), fig_washing_pca(panel, out),
            fig_selective_disclosure(shares, out)]


if __name__ == "__main__":
    for p in main():
        print(f"-> {p}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_eda_index.py -v`
Expected: PASS (5 passed total)

- [ ] **Step 5: Commit**

```bash
git add experiments/eda_indices.py tests/test_eda_index.py
git commit -m "feat(eda): index EDA figures (ribbons, cartography, trajectories, PCA, selective disclosure)"
```

---

## Self-Review

**Spec coverage (Phase 0 = spec §4 EDA + §7 Phase 0):**
- §4 fig 1 chunk lattice+CDF → Task 3 `fig_token_lattice` (histogram + quantile lines; CDF twin-axis folded into histogram for simplicity — acceptable; can add twin-axis later if desired).
- §4 fig 2 coverage cartography → Task 3 `fig_coverage`. ✓
- §4 fig 3 label composition + entropy → Task 3 `fig_labels` + Task 2 `spec_level_distribution`. ✓
- §4 fig 4 noise retention → Task 3 `fig_noise_retention`. ✓
- §4 fig 5 ribbons → Task 5 `fig_ribbons`. ✓
- §4 fig 6 CTI cartography → Task 5 `fig_cti_cartography`. ✓
- §4 fig 7 CTI trajectories (LineCollection) → Task 5 `fig_cti_trajectories`. ✓
- §4 fig 8 washing-space PCA (eigh) → Task 5 `fig_washing_pca` + Task 4 `manual_pca`. ✓ (SBS column added in Phase 4 per spec.)
- §4 fig 9 selective disclosure → Task 5 `fig_selective_disclosure`. ✓
- style module → Task 1. ✓

**Placeholder scan:** none — all steps contain full code.

**Type consistency:** `washing_feature_matrix` returns `(X, rows, feats)` consumed identically in `fig_washing_pca`; `manual_pca(X, n)` → `(scores, loadings, evr)` consumed identically; `trajectory_deltas` dict shape `{bank: {years, values, deltas}}` consumed identically; `quantile_ribbons` columns `q10/q25/median/q75/q90/mean` consumed identically. ✓

**Note for executor:** Tasks 3 and 5 each append a smoke test to a test file created in an earlier task (Tasks 2 and 4 respectively); apply edits in task order.
