# Phase 1 — Embedding washing signals (SBS / BRI) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add unsupervised embedding-based washing signal(s) over commitment chunks and test RQ4 against the rubric-based CTI.

> **UPDATE 2026-06-19 (post-implementation, empirical):** SBS (Substance Backing Score) was
> implemented and tested but **dropped** after running on full data — raw cosine is saturated by
> sentence-embedding anisotropy (mean 0.64, std 0.056) and stays null vs every index even after
> mean-centering; topical similarity ≠ evidential backing. **Mean-centering** of embeddings is now
> standard. **BRI (Boilerplate Reuse Index)** is the kept signal. Reframed RQ4 result (n=45):
> Spearman(CTI, BRI) = **−0.353 (p=0.018)**, Spearman(NAR, BRI) = **+0.300 (p=0.045)** — boilerplate
> = shared NAMED programs (specific, low cheap-talk), not vague cheap-talk. Final code reflects this
> (commit c8a2414); Tasks 1–2 below describe the original SBS+BRI build for history.

**Architecture:** Pure compute (cosine geometry, SBS/BRI aggregation) lives in `src/esgwash/indices/alignment.py`, unit-tested on toy embeddings (no model download). A driver `experiments/embedding_signals.py` loads `outputs/cti/*/*/classified.parquet`, embeds commitment chunks ONCE with the existing `SentenceEmbedder` (bkai vietnamese-bi-encoder, normalized embeddings → cosine = dot product), computes per-(bank,year) SBS/BRI, merges with panel.csv, and reports Spearman correlations for RQ4. Embedding is the only GPU/heavy step and happens in one pass (Kaggle end-to-end).

**Tech Stack:** Python 3.13, numpy, pandas, scipy.stats (spearmanr), sentence-transformers (existing `esgwash.corpus.sentence_embedder.SentenceEmbedder`), pytest.

## Global Constraints

- No external labels; signals derived only from report text. SBS/BRI use only cosine + max + mean (no learned weights).
- Embeddings are L2-normalized (SentenceEmbedder already passes `normalize_embeddings=True`), so cosine similarity == dot product. Compute helpers assume normalized rows.
- Figures (if any) English, DejaVu Sans, PNG dpi 140 into experiments/figures/; reuse `esgwash.eda.style`.
- Driver scripts bootstrap sys.path with repo src/ (`sys.path.insert(0, .../ "src")`) for Kaggle.
- Do NOT modify classified.parquet/panel.csv contracts; the driver writes NEW files only (experiments/panel/embedding_signals.csv).
- ESG-commitment chunk = `is_commitment==1 AND (is_env OR is_soc OR is_gov)`; spec_level in {0,1,2}. Match the CTI denominator definition in src/esgwash/indices/cti.py.
- Tests flat in tests/, pytest, import `from esgwash...`.

## File Structure

- Create `src/esgwash/indices/alignment.py` — pure compute: `pairwise_max_cosine`, `substance_backing_score`, `boilerplate_reuse_index`, `signals_per_panel`.
- Create `tests/test_alignment.py` — unit tests on toy normalized vectors.
- Create `experiments/embedding_signals.py` — driver: embed real commitment chunks, compute panel signals, Spearman vs CTI, write CSV + console report.

Verified schemas: `classified.parquet` has `bank, year, content_text, is_env, is_soc, is_gov, is_commitment, spec_level`. `panel.csv` has `bank, year, cti, nar, qdr, n_commit`. `SentenceEmbedder().embed(list[str]) -> np.ndarray` (normalized rows).

---

### Task 1: Pure cosine + SBS/BRI compute helpers

**Files:**
- Create: `src/esgwash/indices/alignment.py`
- Test: `tests/test_alignment.py`

**Interfaces:**
- Consumes: numpy arrays of L2-normalized embeddings.
- Produces:
  - `pairwise_max_cosine(A: np.ndarray, B: np.ndarray) -> np.ndarray` — for each row of A, the max dot-product against rows of B; shape `(len(A),)`. If B empty, returns zeros of shape `(len(A),)`.
  - `substance_backing_score(emb: np.ndarray, spec_level: np.ndarray) -> tuple[float, np.ndarray]` — over one (bank,year)'s commitment-chunk embeddings: for each vague chunk (spec_level==0), backing = max cosine to quantified chunks (spec_level==2); 0.0 if no quantified chunk. Returns `(mean_backing_over_vague, per_vague_backing_array)`. If no vague chunks, returns `(float("nan"), empty array)`.
  - `boilerplate_reuse_index(emb: np.ndarray, banks: np.ndarray) -> tuple[float, np.ndarray]` — for each chunk, max cosine to chunks whose bank differs; BRI = mean over all chunks. Single-bank input → `(float("nan"), zeros)`. Returns `(mean, per_chunk_array)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_alignment.py
import numpy as np
from esgwash.indices import alignment as al


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_pairwise_max_cosine_picks_nearest():
    A = _unit([[1, 0], [0, 1]])
    B = _unit([[1, 0.01], [-1, 0]])
    out = al.pairwise_max_cosine(A, B)
    assert out.shape == (2,)
    assert out[0] > 0.99            # [1,0] aligns with [1,0.01]
    assert abs(out[1]) < 0.2        # [0,1] orthogonal to both


def test_pairwise_max_cosine_empty_B():
    A = _unit([[1, 0]])
    out = al.pairwise_max_cosine(A, np.empty((0, 2)))
    assert out.shape == (1,) and out[0] == 0.0


def test_sbs_vague_far_from_quantified_is_low():
    # vague chunk orthogonal to the single quantified chunk -> backing ~0
    emb = _unit([[1, 0], [0, 1]])
    spec = np.array([0, 2])
    sbs, per = al.substance_backing_score(emb, spec)
    assert per.shape == (1,)
    assert sbs < 0.2


def test_sbs_vague_near_quantified_is_high():
    emb = _unit([[1, 0.02], [1, 0]])
    spec = np.array([0, 2])
    sbs, _ = al.substance_backing_score(emb, spec)
    assert sbs > 0.99


def test_sbs_no_quantified_gives_zero_backing():
    emb = _unit([[1, 0], [0, 1]])
    spec = np.array([0, 0])
    sbs, per = al.substance_backing_score(emb, spec)
    assert sbs == 0.0 and list(per) == [0.0, 0.0]


def test_sbs_no_vague_is_nan():
    emb = _unit([[1, 0]])
    spec = np.array([2])
    sbs, per = al.substance_backing_score(emb, spec)
    assert np.isnan(sbs) and per.size == 0


def test_bri_identical_other_bank_text_is_high():
    emb = _unit([[1, 0], [1, 0.01], [0, 1]])
    banks = np.array(["a", "b", "a"])
    bri, per = al.boilerplate_reuse_index(emb, banks)
    assert per.shape == (3,)
    assert per[0] > 0.99            # a's [1,0] matches b's [1,0.01]
    assert 0.0 <= bri <= 1.0001


def test_bri_single_bank_is_nan():
    emb = _unit([[1, 0], [0, 1]])
    banks = np.array(["a", "a"])
    bri, per = al.boilerplate_reuse_index(emb, banks)
    assert np.isnan(bri)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_alignment.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'esgwash.indices.alignment'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/esgwash/indices/alignment.py
"""Embedding washing signals (RQ4). Assumes L2-normalized embeddings so
cosine == dot product. Pure geometry: cosine + max + mean, no learned weights.

SBS (Substance Backing Score): per (bank,year), how well VAGUE commitments
(spec_level=0) are backed by a nearby QUANTIFIED commitment (spec_level=2).
Low SBS = vague claims float free of evidence -> corroborates high CTI.

BRI (Boilerplate Reuse Index): how similar each commitment is to commitments
of OTHER banks. High BRI = recycled generic language = cheap-talk signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def pairwise_max_cosine(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.shape[0] == 0:
        return np.zeros(0)
    if B.shape[0] == 0:
        return np.zeros(A.shape[0])
    sims = A @ B.T                      # rows normalized -> cosine
    return sims.max(axis=1)


def substance_backing_score(emb: np.ndarray, spec_level: np.ndarray):
    emb = np.asarray(emb, dtype=float)
    spec_level = np.asarray(spec_level)
    vague = emb[spec_level == 0]
    quant = emb[spec_level == 2]
    if vague.shape[0] == 0:
        return float("nan"), np.zeros(0)
    if quant.shape[0] == 0:
        per = np.zeros(vague.shape[0])
        return 0.0, per
    per = pairwise_max_cosine(vague, quant)
    return float(per.mean()), per


def boilerplate_reuse_index(emb: np.ndarray, banks: np.ndarray):
    emb = np.asarray(emb, dtype=float)
    banks = np.asarray(banks)
    n = emb.shape[0]
    per = np.zeros(n)
    if np.unique(banks).size < 2:
        return float("nan"), per
    for i in range(n):
        other = emb[banks != banks[i]]
        per[i] = pairwise_max_cosine(emb[i:i + 1], other)[0]
    return float(per.mean()), per


def signals_per_panel(df: pd.DataFrame, emb: np.ndarray) -> pd.DataFrame:
    """df = ESG-commitment chunks (cols bank, year, spec_level) aligned row-wise
    with emb. BRI uses same-year cross-bank comparison. -> bank, year, sbs, bri."""
    rows = []
    for year, g_year in df.groupby("year"):
        idx_year = g_year.index.to_numpy()
        emb_year = emb[idx_year]
        banks_year = g_year["bank"].to_numpy()
        _, bri_per = boilerplate_reuse_index(emb_year, banks_year)
        bri_series = pd.Series(bri_per, index=idx_year)
        for bank, g in g_year.groupby("bank"):
            idx = g.index.to_numpy()
            sbs, _ = substance_backing_score(emb[idx], g["spec_level"].to_numpy())
            rows.append({"bank": bank, "year": year, "sbs": sbs,
                         "bri": float(bri_series.loc[idx].mean())})
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_alignment.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add src/esgwash/indices/alignment.py tests/test_alignment.py
git commit -m "feat(indices): SBS/BRI embedding washing signals (pure compute)"
```

---

### Task 2: Driver — embed real chunks, compute signals, RQ4 correlation

**Files:**
- Create: `experiments/embedding_signals.py`
- Test: `tests/test_alignment.py` (append a driver-level test using a fake embedder, no model download)

**Interfaces:**
- Consumes: `esgwash.indices.alignment.signals_per_panel`; `esgwash.corpus.sentence_embedder.SentenceEmbedder`.
- Produces: `compute(classified: pd.DataFrame, embedder) -> pd.DataFrame` (bank, year, sbs, bri) and `main(out_dir="experiments/panel") -> pd.DataFrame`. `main` loads classified + panel.csv, embeds ESG-commitment chunks via SentenceEmbedder, writes `embedding_signals.csv`, prints Spearman(CTI, 1-SBS) and Spearman(CTI, BRI). `compute` takes an injectable `embedder` exposing `.embed(list[str]) -> np.ndarray` so tests pass a fake.

- [ ] **Step 1: Write the failing test** (append to tests/test_alignment.py)

```python
def test_compute_with_fake_embedder():
    import importlib.util, sys
    from pathlib import Path
    import pandas as pd
    spec = importlib.util.spec_from_file_location(
        "embedding_signals", Path(__file__).resolve().parents[1] / "experiments" / "embedding_signals.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["embedding_signals"] = mod
    spec.loader.exec_module(mod)

    # 2 banks, 1 year; bank a: vague + quantified aligned; bank b: vague alone
    clf = pd.DataFrame({
        "bank": ["a", "a", "b"], "year": [2023, 2023, 2023],
        "content_text": ["green pledge", "green 5000bn", "vague aspiration"],
        "is_env": [1, 1, 1], "is_soc": [0, 0, 0], "is_gov": [0, 0, 0],
        "is_commitment": [1, 1, 1], "spec_level": [0, 2, 0],
    })

    class FakeEmb:
        def embed(self, texts):
            import numpy as np
            m = {"green pledge": [1, 0], "green 5000bn": [1, 0.01],
                 "vague aspiration": [0, 1]}
            v = np.array([m[t] for t in texts], dtype=float)
            return v / np.linalg.norm(v, axis=1, keepdims=True)

    out = mod.compute(clf, FakeEmb())
    a = out[out["bank"] == "a"].iloc[0]
    b = out[out["bank"] == "b"].iloc[0]
    assert a["sbs"] > 0.99          # a's vague backed by its quantified
    assert b["sbs"] == 0.0          # b has no quantified chunk
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_alignment.py::test_compute_with_fake_embedder -v`
Expected: FAIL (FileNotFoundError for experiments/embedding_signals.py)

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/embedding_signals.py
"""Embed commitment chunks, compute SBS/BRI per (bank,year), test RQ4 vs CTI."""
from __future__ import annotations

import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from scipy.stats import spearmanr

from esgwash.indices.alignment import signals_per_panel

PILLARS = ["is_env", "is_soc", "is_gov"]


def _esg_commit(clf: pd.DataFrame) -> pd.DataFrame:
    esg = clf[PILLARS].max(axis=1).astype(bool)
    return clf[(clf["is_commitment"] == 1) & esg].reset_index(drop=True)


def compute(classified: pd.DataFrame, embedder) -> pd.DataFrame:
    df = _esg_commit(classified)
    emb = embedder.embed(df["content_text"].astype(str).tolist())
    return signals_per_panel(df, emb)


def _load_classified() -> pd.DataFrame:
    files = sorted(glob.glob("outputs/cti/*/*/classified.parquet"))
    frames = [pd.read_parquet(f) for f in files]
    return pd.concat([f for f in frames if not f.empty], ignore_index=True)


def main(out_dir: str = "experiments/panel") -> pd.DataFrame:
    from esgwash.corpus.sentence_embedder import SentenceEmbedder
    clf = _load_classified()
    sig = compute(clf, SentenceEmbedder())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sig.to_csv(out / "embedding_signals.csv", index=False)

    panel = pd.read_csv(out / "panel.csv").merge(sig, on=["bank", "year"], how="left")
    panel.to_csv(out / "panel_with_signals.csv", index=False)
    valid = panel.dropna(subset=["sbs", "cti"])
    rho_s, p_s = spearmanr(valid["cti"], 1 - valid["sbs"])
    vb = panel.dropna(subset=["bri", "cti"])
    rho_b, p_b = spearmanr(vb["cti"], vb["bri"])
    print(f"RQ4 convergent validity (n={len(valid)}):")
    print(f"  Spearman(CTI, 1-SBS) = {rho_s:.3f} (p={p_s:.2e})")
    print(f"  Spearman(CTI, BRI)   = {rho_b:.3f} (p={p_b:.2e})")
    print(f"-> {out/'embedding_signals.csv'}")
    return panel


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_alignment.py -v`
Expected: PASS (9 passed total)

- [ ] **Step 5: Commit**

```bash
git add experiments/embedding_signals.py tests/test_alignment.py
git commit -m "feat(indices): embedding-signals driver + RQ4 Spearman vs CTI"
```

---

## Self-Review

**Spec coverage (spec §5 M1+M2, §3 RQ4):**
- M1 SBS (claim–evidence alignment gap) → Task 1 `substance_backing_score`. ✓
- M2 BRI (boilerplate reuse) → Task 1 `boilerplate_reuse_index`. ✓
- Per-(bank,year) panel signals → Task 1 `signals_per_panel`. ✓
- Embed real chunks once, merge panel, RQ4 Spearman → Task 2 driver. ✓
- Reuse existing SentenceEmbedder (normalized) → Task 2. ✓
- KMeans clustering / washing-themes (spec §5 M2) → deferred to a follow-up task (Phase 1b) to keep this plan's first cut focused on the headline RQ4 signals; noted here so it is not lost.

**Placeholder scan:** none — all steps contain full code.

**Type consistency:** `pairwise_max_cosine(A,B)->ndarray` consumed by SBS/BRI; `substance_backing_score->(float, ndarray)` and `boilerplate_reuse_index->(float, ndarray)` consumed by `signals_per_panel`; `signals_per_panel(df, emb)->DataFrame[bank,year,sbs,bri]` consumed by driver `compute`; `compute(classified, embedder)->DataFrame` used in `main` and test. ✓

**Note for executor:** Task 2 appends a test to tests/test_alignment.py created in Task 1; apply in order. The driver's `compute` takes an injectable embedder so the test never downloads a model.
