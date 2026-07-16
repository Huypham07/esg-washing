# Phase 3 — Interpretable lexical baseline + cross-lingual transfer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add a CPU-only interpretable lexical baseline (TF-IDF + LogisticRegression) that (a) reports per-head F1, (b) shows top discriminative tokens per head (explainability without a transformer), and (c) quantifies lexical cross-lingual transfer (train EN → test VI vs in-language) to motivate translate-train / a multilingual encoder.

**Architecture:** Pure compute in `src/esgwash/models/interpret.py` (fit TF-IDF+LR, extract top tokens, cross-lingual macro-F1), unit-tested on toy frames. Driver `experiments/baseline_interpret.py` runs on the local gold parquets (which carry both `text` = VI and `text_en` = EN), writes a transfer table CSV + per-head token figures, and prints results.

> **Scope note (transformer rigor deferred to Kaggle):** The transformer encoder comparison
> (PhoBERT vs XLM-R), evaluation of the trained HF models (`huypham71/esg-*` are private — need a
> token), and SHAP/attention on transformers require GPU + auth and run in the Kaggle environment
> (`experiments/eval_models.py`, `esgwash.models.baselines.xlmr_zero_shot` already exist for that).
> This plan covers the CPU-runnable, immediately-showable slice.

**Tech Stack:** Python 3.13, scikit-learn (TfidfVectorizer, LogisticRegression, f1_score), pandas, numpy, matplotlib, pytest.

## Global Constraints

- CPU only; no transformer training, no HF download, no Java.
- Gold parquets: `data/topic_{train,test}.parquet` (cols `text, text_en, env, soc, gov, sources`), `data/commitment_{train,test}.parquet` (cols `text, text_en, commitment, source`). `text` = Vietnamese, `text_en` = English.
- Figures English, DejaVu Sans, PNG dpi 140 into experiments/figures/; reuse `esgwash.eda.style`.
- Driver bootstraps sys.path with repo src/. Writes NEW files only (experiments/eval/transfer.csv, top_tokens.csv).
- Reuse the same TF-IDF+LR config as `esgwash.models.baselines.tfidf_lr_baseline` (ngram (1,2), max_features 50000, sublinear_tf; LR max_iter 2000, class_weight balanced) so numbers are comparable.
- Tests flat in tests/, pytest, import `from esgwash...`.

## File Structure

- Create `src/esgwash/models/interpret.py` — `fit_head`, `top_tokens`, `cross_lingual_macro_f1`.
- Create `experiments/baseline_interpret.py` — driver: transfer table + token figures.
- Create `tests/test_interpret.py`.

---

### Task 1: Interpretable baseline compute helpers

**Files:**
- Create: `src/esgwash/models/interpret.py`
- Test: `tests/test_interpret.py`

**Interfaces:**
- Produces:
  - `fit_head(train_df, head, text_col="text", seed=42) -> Pipeline` — TF-IDF+LR pipeline fit on rows where `head` is non-null.
  - `top_tokens(train_df, head, text_col="text", k=15, seed=42) -> list[tuple[str, float]]` — top-k tokens by positive LR coefficient (descending).
  - `cross_lingual_macro_f1(train_df, test_df, heads, train_col, test_col, seed=42) -> float` — mean per-head binary F1 training on `train_col`, testing on `test_col`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_interpret.py
import numpy as np
import pandas as pd
from esgwash.models import interpret as it


def _toy():
    pos = ["green energy target reduce emissions"] * 20
    neg = ["the bank board meeting quarterly report"] * 20
    df = pd.DataFrame({
        "text": pos + neg,
        "text_en": pos + neg,
        "env": [1] * 20 + [0] * 20,
    })
    return df


def test_top_tokens_returns_sorted_k():
    toks = it.top_tokens(_toy(), "env", text_col="text", k=5)
    assert len(toks) == 5
    weights = [w for _, w in toks]
    assert weights == sorted(weights, reverse=True)
    # an env-positive word should surface
    assert any(t in {"green", "energy", "emissions", "reduce", "target"}
               for t, _ in toks)


def test_cross_lingual_macro_f1_in_range():
    df = _toy()
    f1 = it.cross_lingual_macro_f1(df, df, ["env"], "text", "text", seed=0)
    assert 0.0 <= f1 <= 1.0
    assert f1 > 0.8   # toy is linearly separable


def test_fit_head_skips_nan_rows():
    df = _toy()
    df.loc[0, "env"] = np.nan       # one NaN label must be dropped, no crash
    pipe = it.fit_head(df, "env", "text")
    assert pipe.predict(["green energy emissions"]).shape == (1,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_interpret.py -v`
Expected: FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/esgwash/models/interpret.py
"""Interpretable lexical baseline (CPU): TF-IDF + LogisticRegression per head.

- top_tokens: which words drive each classifier (explainability, no transformer).
- cross_lingual_macro_f1: train on one language column, test on another (text=VI,
  text_en=EN) to quantify lexical cross-lingual transfer. Same TF-IDF+LR config as
  esgwash.models.baselines.tfidf_lr_baseline so numbers are comparable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline


def fit_head(train_df: pd.DataFrame, head: str, text_col: str = "text", seed: int = 42):
    tr = train_df.dropna(subset=[head])
    pipe = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), max_features=50_000, sublinear_tf=True),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))
    pipe.fit(tr[text_col].astype(str), tr[head].astype(int))
    return pipe


def top_tokens(train_df: pd.DataFrame, head: str, text_col: str = "text",
               k: int = 15, seed: int = 42):
    pipe = fit_head(train_df, head, text_col, seed)
    vec = pipe.named_steps["tfidfvectorizer"]
    clf = pipe.named_steps["logisticregression"]
    names = np.asarray(vec.get_feature_names_out())
    coef = clf.coef_[0]
    order = np.argsort(coef)[::-1][:k]
    return [(str(names[i]), float(coef[i])) for i in order]


def cross_lingual_macro_f1(train_df: pd.DataFrame, test_df: pd.DataFrame, heads,
                           train_col: str, test_col: str, seed: int = 42) -> float:
    f1s = []
    for h in heads:
        tr = train_df.dropna(subset=[h])
        te = test_df.dropna(subset=[h])
        if tr.empty or te.empty:
            continue
        pipe = fit_head(tr, h, train_col, seed)
        f1s.append(f1_score(te[h].astype(int), pipe.predict(te[test_col].astype(str))))
    return float(np.mean(f1s)) if f1s else 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_interpret.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/esgwash/models/interpret.py tests/test_interpret.py
git commit -m "feat(models): interpretable TF-IDF+LR baseline (top tokens, cross-lingual F1)"
```

---

### Task 2: Driver — transfer table + token figures

**Files:**
- Create: `experiments/baseline_interpret.py`
- Test: `tests/test_interpret.py` (append a driver smoke test)

**Interfaces:**
- Produces: `transfer_table(train_df, test_df, heads) -> pd.DataFrame` (rows = configs VI→VI / EN→VI / EN→EN, column `macro_f1`); `main(out_fig="experiments/figures", out_tab="experiments/eval") -> dict` — loads topic+commitment gold, builds transfer tables for both, writes `transfer.csv` + `top_tokens.csv`, renders one token bar figure per task, prints the tables.

- [ ] **Step 1: Write the failing test** (append to tests/test_interpret.py)

```python
def test_transfer_table_has_three_configs():
    import importlib.util, sys
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "baseline_interpret", Path(__file__).resolve().parents[1] / "experiments" / "baseline_interpret.py")
    mod = importlib.util.module_from_spec(spec); sys.modules["baseline_interpret"] = mod
    spec.loader.exec_module(mod)
    df = _toy()
    out = mod.transfer_table(df, df, ["env"])
    assert list(out["config"]) == ["VI->VI", "EN->VI", "EN->EN"]
    assert out["macro_f1"].between(0, 1).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_interpret.py::test_transfer_table_has_three_configs -v`
Expected: FAIL (FileNotFoundError for experiments/baseline_interpret.py).

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/baseline_interpret.py
"""Phase 3 (CPU): interpretable lexical baseline + cross-lingual transfer table.
Transformer/encoder comparison + transformer SHAP run on Kaggle (GPU/auth)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from esgwash.eda.style import PALETTE, SCORE_CMAP, apply_rcparams, style_axes
from esgwash.models.interpret import cross_lingual_macro_f1, top_tokens

_CONFIGS = [("VI->VI", "text", "text"), ("EN->VI", "text_en", "text"),
            ("EN->EN", "text_en", "text_en")]


def transfer_table(train_df, test_df, heads) -> pd.DataFrame:
    rows = []
    for name, tr_col, te_col in _CONFIGS:
        f1 = cross_lingual_macro_f1(train_df, test_df, heads, tr_col, te_col)
        rows.append({"config": name, "macro_f1": round(f1, 4)})
    return pd.DataFrame(rows)


def _fig_tokens(train_df, heads, title, out_fig: Path, fname: str) -> Path:
    fig, axes = plt.subplots(1, len(heads), figsize=(5 * len(heads), 5),
                             facecolor=PALETTE["paper"])
    if len(heads) == 1:
        axes = [axes]
    for ax, h in zip(axes, heads):
        toks = top_tokens(train_df, h, "text", k=12)[::-1]
        labels = [t for t, _ in toks]
        vals = [w for _, w in toks]
        style_axes(ax, f"{h}", "Top TF-IDF+LR tokens (Vietnamese).")
        ax.barh(range(len(labels)), vals,
                color=SCORE_CMAP(np.linspace(0.2, 0.95, len(labels))),
                edgecolor=PALETTE["paper"])
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.set_xlabel("LR coefficient")
    fig.suptitle(title, x=0.02, ha="left", fontweight="bold", color=PALETTE["ink"])
    fig.tight_layout()
    out_fig.mkdir(parents=True, exist_ok=True)
    p = out_fig / fname
    fig.savefig(p, bbox_inches="tight", facecolor=PALETTE["paper"])
    plt.close(fig)
    return p


def main(out_fig: str = "experiments/figures", out_tab: str = "experiments/eval") -> dict:
    apply_rcparams()
    topic_tr = pd.read_parquet("data/topic_train.parquet")
    topic_te = pd.read_parquet("data/topic_test.parquet")
    com_tr = pd.read_parquet("data/commitment_train.parquet")
    com_te = pd.read_parquet("data/commitment_test.parquet")
    topic_heads = [h for h in ["env", "soc", "gov"] if h in topic_tr.columns]

    t_topic = transfer_table(topic_tr, topic_te, topic_heads).assign(task="topic")
    t_com = transfer_table(com_tr, com_te, ["commitment"]).assign(task="commitment")
    transfer = pd.concat([t_topic, t_com], ignore_index=True)

    tab = Path(out_tab); tab.mkdir(parents=True, exist_ok=True)
    transfer.to_csv(tab / "transfer.csv", index=False)

    tok_rows = []
    for task, df, heads in [("topic", topic_tr, topic_heads), ("commitment", com_tr, ["commitment"])]:
        for h in heads:
            for tok, w in top_tokens(df, h, "text", k=15):
                tok_rows.append({"task": task, "head": h, "token": tok, "coef": round(w, 4)})
    pd.DataFrame(tok_rows).to_csv(tab / "top_tokens.csv", index=False)

    fig = Path(out_fig)
    _fig_tokens(topic_tr, topic_heads, "Topic classifier — top tokens per pillar", fig, "tokens_topic.png")
    _fig_tokens(com_tr, ["commitment"], "Commitment classifier — top tokens", fig, "tokens_commitment.png")

    print("Cross-lingual transfer (TF-IDF+LR macro-F1):")
    print(transfer.to_string(index=False))
    return {"transfer": transfer}


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_interpret.py -v`
Expected: PASS (4 passed total)

- [ ] **Step 5: Commit**

```bash
git add experiments/baseline_interpret.py tests/test_interpret.py
git commit -m "feat(models): cross-lingual transfer table + per-head token figures"
```

---

## Self-Review

**Spec coverage (spec §6 model rigor — CPU slice):**
- Baseline F1 per head → Task 1 `cross_lingual_macro_f1` (VI→VI config) + Task 2 table. ✓
- Token-level explainability → Task 1 `top_tokens` + Task 2 figures. ✓
- Cross-lingual transfer (train EN→test VI) → Task 2 `transfer_table` EN→VI row. ✓
- Transformer encoder comparison / transformer SHAP → DEFERRED to Kaggle (documented scope note; existing eval_models.py / baselines.xlmr_zero_shot cover it). ✓ (intentional, not a gap)

**Placeholder scan:** none.

**Type consistency:** `fit_head->Pipeline` used by `top_tokens`/`cross_lingual_macro_f1`; `top_tokens->list[(str,float)]` consumed by `_fig_tokens`; `transfer_table->DataFrame[config,macro_f1]` consumed by `main`. ✓

**Note for executor:** Task 2 appends a test to tests/test_interpret.py from Task 1; the `_toy()` helper is reused.
