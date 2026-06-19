# Phase 2 — Internal numerical axis (say-do gap) + chunk-level SHAP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build the internal numerical axis from the rubric's own extracted figures (no new extraction model), a per-pillar say-do gap, and a chunk-level GradientBoosting model with SHAP explaining what textual features drive specificity (RQ5).

**Architecture:** Pure compute (parse rubric figures, categorize, figure table, per-pillar say-do, chunk feature engineering) lives in `src/esgwash/indices/figures_extract.py` and `src/esgwash/models/spec_features.py`, unit-tested on toy DataFrames. A driver `experiments/saydo_shap.py` loads `outputs/cti/*/*/classified.parquet`, trains a GradientBoostingClassifier on chunk features (stratified CV macro-F1), renders a SHAP summary plot, a per-pillar say-do heatmap, and a figure-type bar — reusing `esgwash.eda.style`. Panel-level outputs are descriptive (N=45) with an explicit small-N caveat; the ML model is chunk-level (large N) where SHAP is valid.

**Tech Stack:** Python 3.13, pandas, numpy, scikit-learn (GradientBoostingClassifier, StratifiedKFold, f1_score), shap>=0.44, matplotlib, pytest.

## Global Constraints

- No external labels. Figures come ONLY from the rubric's already-extracted `figure`/`action_or_event` (already filtered by `verify_rubric`); NO new number-extraction model.
- The chunk-level model MUST NOT use bank/ticker/doc identifiers as features (data-leakage lesson from the reference SHAP notebook). Features are structural/textual only.
- ESG-commitment chunk = `is_commitment==1 AND (is_env OR is_soc OR is_gov)`; `spec_level` in {0,1,2}.
- say-do gap per (bank,year,pillar) = `CTI_pillar - QDR_pillar` where `CTI_pillar = share(spec_level==0)` and `QDR_pillar = share(spec_level==2)` among commitment chunks tagged with that pillar; range [-1,1].
- Panel-level (N=45) outputs are descriptive only — figures/captions must say "exploratory"; no predictive claim at panel level. The GBM is chunk-level.
- Figures English, DejaVu Sans, PNG dpi 140 into experiments/figures/; reuse `esgwash.eda.style`.
- Driver bootstraps sys.path with repo src/. Writes NEW files only (experiments/panel/figure_types.csv, say_do.csv); does not modify existing contracts.
- Tests flat in tests/, pytest, import `from esgwash...`.

## File Structure

- Create `src/esgwash/indices/figures_extract.py` — `parse_figures`, `categorize_action`, `figure_table`, `pillar_say_do`.
- Create `src/esgwash/models/spec_features.py` — `chunk_features` (X, y) for the spec_level model.
- Create `experiments/saydo_shap.py` — driver: GBM CV + SHAP plot + say-do heatmap + figure-type bar.
- Create `tests/test_figures_extract.py`, `tests/test_spec_features.py`.

Verified: `spec_rubric` JSON = `{"items": [{"action_or_event", "figure", "is_quantified", "attributable_to_actor", "is_concrete_action"}], "has_baseline_or_timeline", "reason"}`. `classified.parquet` has `content_text, token_count, char_count, chunk_index, doc_id, bank, year, p_env, p_soc, p_gov, is_env, is_soc, is_gov, p_commitment, is_commitment, spec_level, spec_rubric`.

---

### Task 1: Rubric figure extraction + per-pillar say-do

**Files:**
- Create: `src/esgwash/indices/figures_extract.py`
- Test: `tests/test_figures_extract.py`

**Interfaces:**
- Produces:
  - `parse_figures(spec_rubric: str | None) -> list[dict]` — JSON string -> list of `{"action": str, "figure": str}` for items where `is_quantified` is true and `figure` is non-empty. Bad/empty/None input -> `[]`.
  - `categorize_action(action: str) -> str` — keyword map (priority order) -> one of `green_credit, emissions, energy, social, trees, training, other`.
  - `figure_table(classified: pd.DataFrame) -> pd.DataFrame` — over ESG-commitment chunks: explode rubric figures, categorize, count -> columns `bank, year, type, n`.
  - `pillar_say_do(classified: pd.DataFrame) -> pd.DataFrame` — per (bank, year, pillar in env/soc/gov): `cti_p, qdr_p, say_do (=cti_p-qdr_p), n`. Only commitment chunks tagged with that pillar.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_figures_extract.py
import json
import pandas as pd
from esgwash.indices import figures_extract as fx


def test_parse_figures_keeps_quantified_with_figure():
    rub = json.dumps({"items": [
        {"action_or_event": "dư nợ tín dụng xanh", "figure": "74.000 tỷ",
         "is_quantified": True, "attributable_to_actor": True, "is_concrete_action": True},
        {"action_or_event": "nâng cao năng lực", "figure": None,
         "is_quantified": False, "attributable_to_actor": True, "is_concrete_action": False},
    ]})
    out = fx.parse_figures(rub)
    assert len(out) == 1
    assert out[0]["action"] == "dư nợ tín dụng xanh" and out[0]["figure"] == "74.000 tỷ"


def test_parse_figures_bad_input():
    assert fx.parse_figures(None) == []
    assert fx.parse_figures("not json") == []
    assert fx.parse_figures(json.dumps({"items": []})) == []


def test_categorize_action_priority():
    assert fx.categorize_action("dư nợ tín dụng xanh cho vay") == "green_credit"
    assert fx.categorize_action("giảm phát thải khí nhà kính") == "emissions"
    assert fx.categorize_action("lắp điện mặt trời") == "energy"
    assert fx.categorize_action("trồng 330.000 cây xanh") == "trees"
    assert fx.categorize_action("đào tạo cán bộ nhân viên") == "training"
    assert fx.categorize_action("ủng hộ quỹ an sinh xã hội") == "social"
    assert fx.categorize_action("một việc gì đó") == "other"


def test_figure_table_counts_by_type():
    rub = json.dumps({"items": [
        {"action_or_event": "dư nợ tín dụng xanh", "figure": "5 tỷ",
         "is_quantified": True, "attributable_to_actor": True, "is_concrete_action": True}]})
    clf = pd.DataFrame({
        "bank": ["a"], "year": [2023], "is_env": [1], "is_soc": [0], "is_gov": [0],
        "is_commitment": [1], "spec_level": [2], "spec_rubric": [rub]})
    out = fx.figure_table(clf)
    row = out.iloc[0]
    assert row["type"] == "green_credit" and row["n"] == 1


def test_pillar_say_do():
    # env: 2 vague + 1 quantified -> cti_p=2/3, qdr_p=1/3, say_do=1/3
    clf = pd.DataFrame({
        "bank": ["a", "a", "a"], "year": [2023, 2023, 2023],
        "is_env": [1, 1, 1], "is_soc": [0, 0, 0], "is_gov": [0, 0, 0],
        "is_commitment": [1, 1, 1], "spec_level": [0, 0, 2], "spec_rubric": [None, None, None]})
    out = fx.pillar_say_do(clf)
    env = out[out["pillar"] == "env"].iloc[0]
    assert env["n"] == 3
    assert abs(env["cti_p"] - 2/3) < 1e-9 and abs(env["qdr_p"] - 1/3) < 1e-9
    assert abs(env["say_do"] - 1/3) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_figures_extract.py -v`
Expected: FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/esgwash/indices/figures_extract.py
"""Internal numerical axis (RQ5): reuse the specificity rubric's already-extracted
figures (no new extraction model) + per-pillar say-do gap.

say-do gap per (bank,year,pillar) = CTI_pillar - QDR_pillar (share vague minus
share quantified among that pillar's commitment chunks); >0 = says more than does.
"""
from __future__ import annotations

import json

import pandas as pd

PILLARS = ("env", "soc", "gov")
_ESG = [f"is_{p}" for p in PILLARS]

# priority-ordered keyword map (first match wins)
_CATEGORIES = [
    ("green_credit", ("tín dụng xanh", "dư nợ", "cho vay", "trái phiếu xanh", "giải ngân", "tài trợ vốn")),
    ("emissions", ("phát thải", "khí nhà kính", "carbon", "co2", "scope")),
    ("energy", ("năng lượng", "điện mặt trời", "điện gió", "tái tạo", "mw ", " mw")),
    ("trees", ("trồng", "cây xanh", "cây")),
    ("training", ("đào tạo", "tập huấn", "cán bộ", "nhân viên", "nhân sự")),
    ("social", ("ủng hộ", "từ thiện", "an sinh", "học bổng", "giáo dục", "y tế", "cộng đồng")),
]


def parse_figures(spec_rubric):
    if not spec_rubric or not isinstance(spec_rubric, str):
        return []
    try:
        obj = json.loads(spec_rubric)
    except (json.JSONDecodeError, TypeError):
        return []
    out = []
    for it in (obj.get("items") or []):
        fig = it.get("figure")
        if it.get("is_quantified") and fig:
            out.append({"action": str(it.get("action_or_event") or ""), "figure": str(fig)})
    return out


def categorize_action(action: str) -> str:
    low = str(action).lower()
    for name, kws in _CATEGORIES:
        if any(k in low for k in kws):
            return name
    return "other"


def _esg_commit(classified: pd.DataFrame) -> pd.DataFrame:
    esg = classified[_ESG].max(axis=1).astype(bool)
    return classified[(classified["is_commitment"] == 1) & esg]


def figure_table(classified: pd.DataFrame) -> pd.DataFrame:
    sub = _esg_commit(classified)
    rows = []
    for _, r in sub.iterrows():
        for f in parse_figures(r.get("spec_rubric")):
            rows.append({"bank": r["bank"], "year": r["year"],
                         "type": categorize_action(f["action"])})
    if not rows:
        return pd.DataFrame(columns=["bank", "year", "type", "n"])
    df = pd.DataFrame(rows)
    return df.groupby(["bank", "year", "type"]).size().rename("n").reset_index()


def pillar_say_do(classified: pd.DataFrame) -> pd.DataFrame:
    sub = _esg_commit(classified)
    rows = []
    for p in PILLARS:
        g = sub[sub[f"is_{p}"] == 1]
        for (bank, year), gg in g.groupby(["bank", "year"]):
            lv = gg["spec_level"].to_numpy()
            cti_p = float((lv == 0).mean())
            qdr_p = float((lv == 2).mean())
            rows.append({"bank": bank, "year": year, "pillar": p,
                         "cti_p": cti_p, "qdr_p": qdr_p,
                         "say_do": cti_p - qdr_p, "n": int(len(gg))})
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_figures_extract.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/esgwash/indices/figures_extract.py tests/test_figures_extract.py
git commit -m "feat(indices): rubric figure extraction + per-pillar say-do gap"
```

---

### Task 2: Chunk-level features for the specificity model

**Files:**
- Create: `src/esgwash/models/spec_features.py`
- Test: `tests/test_spec_features.py`

**Interfaces:**
- Produces:
  - `chunk_features(classified: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]` — over ESG-commitment chunks: returns (X, y) where y = `spec_level` and X has ONLY structural/textual columns (NO bank/year/doc/ticker): `token_count, char_count, word_count, has_digit, n_digit_runs, has_year, pct_digit_chars, p_env, p_soc, p_gov, p_commitment, rel_position`. `rel_position = chunk_index / max(chunk_index in same doc_id)` (0 if max is 0).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_spec_features.py
import pandas as pd
from esgwash.models import spec_features as sf


def _clf():
    return pd.DataFrame({
        "bank": ["a", "a"], "year": [2023, 2023], "doc_id": ["a_2023", "a_2023"],
        "chunk_index": [0, 4], "content_text": ["green pledge", "giảm 30% phát thải 2030"],
        "token_count": [3, 6], "char_count": [11, 24],
        "is_env": [1, 1], "is_soc": [0, 0], "is_gov": [0, 0],
        "p_env": [0.8, 0.9], "p_soc": [0.1, 0.1], "p_gov": [0.1, 0.1],
        "is_commitment": [1, 1], "p_commitment": [0.7, 0.8], "spec_level": [0, 2]})


def test_feature_columns_have_no_identifiers():
    X, y = sf.chunk_features(_clf())
    for banned in ["bank", "year", "doc_id", "ticker", "chunk_index"]:
        assert banned not in X.columns
    assert list(y) == [0, 2]


def test_digit_and_year_features():
    X, y = sf.chunk_features(_clf())
    r1 = X.iloc[1]   # "giảm 30% phát thải 2030"
    assert r1["has_digit"] == 1 and r1["has_year"] == 1
    assert r1["n_digit_runs"] >= 2
    r0 = X.iloc[0]   # "green pledge"
    assert r0["has_digit"] == 0 and r0["has_year"] == 0


def test_rel_position():
    X, y = sf.chunk_features(_clf())
    # doc max chunk_index = 4 -> positions 0/4=0.0 and 4/4=1.0
    assert X.iloc[0]["rel_position"] == 0.0
    assert X.iloc[1]["rel_position"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_spec_features.py -v`
Expected: FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/esgwash/models/spec_features.py
"""Chunk-level structural features for the spec_level explainer (RQ5 / SHAP).

DELIBERATELY excludes bank/ticker/doc identifiers — encoding them leaks the
target (the reference SHAP notebook hit 100% accuracy via label leakage)."""
from __future__ import annotations

import re

import pandas as pd

PILLARS = ("env", "soc", "gov")
_ESG = [f"is_{p}" for p in PILLARS]
_DIGIT_RUN = re.compile(r"\d+")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")

FEATURE_COLS = ["token_count", "char_count", "word_count", "has_digit",
                "n_digit_runs", "has_year", "pct_digit_chars",
                "p_env", "p_soc", "p_gov", "p_commitment", "rel_position"]


def _esg_commit(classified: pd.DataFrame) -> pd.DataFrame:
    esg = classified[_ESG].max(axis=1).astype(bool)
    return classified[(classified["is_commitment"] == 1) & esg].reset_index(drop=True)


def chunk_features(classified: pd.DataFrame):
    df = _esg_commit(classified)
    text = df["content_text"].astype(str)
    digit_runs = text.apply(lambda s: _DIGIT_RUN.findall(s))
    n_digit_chars = text.apply(lambda s: sum(c.isdigit() for c in s))
    max_idx = df.groupby("doc_id")["chunk_index"].transform("max")
    rel = df["chunk_index"] / max_idx.where(max_idx > 0, 1)
    rel = rel.where(max_idx > 0, 0.0)

    X = pd.DataFrame({
        "token_count": df["token_count"].astype(float),
        "char_count": df["char_count"].astype(float),
        "word_count": text.str.split().apply(len).astype(float),
        "has_digit": digit_runs.apply(lambda r: int(len(r) > 0)),
        "n_digit_runs": digit_runs.apply(len).astype(float),
        "has_year": text.apply(lambda s: int(bool(_YEAR.search(s)))),
        "pct_digit_chars": (n_digit_chars / df["char_count"].clip(lower=1)).astype(float),
        "p_env": df["p_env"].astype(float),
        "p_soc": df["p_soc"].astype(float),
        "p_gov": df["p_gov"].astype(float),
        "p_commitment": df["p_commitment"].astype(float),
        "rel_position": rel.astype(float),
    }, columns=FEATURE_COLS)
    y = df["spec_level"].astype(int)
    return X, y
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_spec_features.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/esgwash/models/spec_features.py tests/test_spec_features.py
git commit -m "feat(models): chunk-level structural features for spec_level explainer"
```

---

### Task 3: Driver — GBM CV + SHAP + say-do/figure figures

**Files:**
- Create: `experiments/saydo_shap.py`
- Test: `tests/test_spec_features.py` (append a driver smoke test)

**Interfaces:**
- Produces: `train_eval(X, y, seed=42) -> dict` (fits GradientBoostingClassifier, returns `{"macro_f1_cv": float, "model": fitted, "n": int}` using StratifiedKFold(min(3, smallest-class-count)) macro-F1); `main(out_dir_fig="experiments/figures", out_dir_tab="experiments/panel", classified=None) -> dict` — loads classified if None, builds features, CV-evaluates, fits, writes SHAP summary `shap_spec_level.png`, say-do heatmap `index_say_do.png`, figure-type bar `figure_types.png`, plus `figure_types.csv` and `say_do.csv`; returns the metrics dict.

- [ ] **Step 1: Write the failing test** (append to tests/test_spec_features.py)

```python
def test_train_eval_runs_on_toy():
    import importlib.util, sys, numpy as np
    from pathlib import Path
    import pandas as pd
    spec = importlib.util.spec_from_file_location(
        "saydo_shap", Path(__file__).resolve().parents[1] / "experiments" / "saydo_shap.py")
    mod = importlib.util.module_from_spec(spec); sys.modules["saydo_shap"] = mod
    spec.loader.exec_module(mod)
    rng = np.random.default_rng(0)
    n = 60
    X = pd.DataFrame({c: rng.random(n) for c in __import__(
        "esgwash.models.spec_features", fromlist=["FEATURE_COLS"]).FEATURE_COLS})
    y = pd.Series(([0, 1, 2] * (n // 3)))
    out = mod.train_eval(X, y)
    assert "macro_f1_cv" in out and 0.0 <= out["macro_f1_cv"] <= 1.0
    assert out["n"] == n
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_spec_features.py::test_train_eval_runs_on_toy -v`
Expected: FAIL (FileNotFoundError for experiments/saydo_shap.py).

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/saydo_shap.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_spec_features.py -v`
Expected: PASS (4 passed total)

- [ ] **Step 5: Commit**

```bash
git add experiments/saydo_shap.py tests/test_spec_features.py
git commit -m "feat(models): GBM+SHAP spec_level explainer + say-do/figure-type figures"
```

---

## Self-Review

**Spec coverage (spec §5 M3, §3 RQ5):**
- Parse rubric figures (no new model) → Task 1 `parse_figures`/`figure_table`. ✓
- Keyword categorization → Task 1 `categorize_action`. ✓
- say-do gap per pillar (CTI_p − QDR_p) → Task 1 `pillar_say_do`. ✓
- Chunk-level features, NO bank/ticker (anti-leakage) → Task 2 `chunk_features`. ✓
- GBM + stratified CV macro-F1 + SHAP summary → Task 3 `train_eval`/`fig_shap`. ✓
- Panel descriptive with exploratory caption → Task 3 `fig_say_do` title says "exploratory". ✓

**Placeholder scan:** none — all steps contain full code.

**Type consistency:** `chunk_features->(X,y)` consumed by `train_eval(X,y)` and `fig_shap(model,X)`; `FEATURE_COLS` shared between spec_features and the driver/test; `pillar_say_do->DataFrame[...say_do]` consumed by `fig_say_do`; `figure_table->DataFrame[bank,year,type,n]` consumed by `fig_figure_types`. ✓

**Note for executor:** Task 3 appends a test to tests/test_spec_features.py created in Task 2; apply in order. `fig_say_do`/`fig_figure_types` return tuples `(Path, DataFrame)` — `main` unpacks them.
