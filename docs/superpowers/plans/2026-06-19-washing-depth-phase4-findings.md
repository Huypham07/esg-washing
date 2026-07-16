# Phase 4 — Unified panel + RQ1–RQ5 findings report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Merge every per-(bank,year) signal (CTI/NAR/QDR, BRI, say-do per pillar) into one master panel, and generate a consolidated `findings.md` that states RQ1–RQ5 with the real numbers and figure references — the paper-ready source content.

**Architecture:** Pure merge/stat helpers in `src/esgwash/indices/panel_master.py` (unit-tested on toy frames). Driver `experiments/build_findings.py` loads the artifacts already produced by Phases 0–3 (`experiments/panel/panel.csv`, `embedding_signals.csv`, `say_do.csv`, `figure_types.csv`, `summary.json`, `experiments/eval/transfer.csv`), recomputes the few cheap correlations, writes `panel_master.csv` + `findings.md`.

> **Scope note:** This phase only consolidates and reports artifacts that already exist on disk
> (plus cheap recomputed correlations). It does NOT retrain models or edit the LaTeX paper
> (per the project's "manage content, don't compile LaTeX" rule). `findings.md` is the source the
> user folds into `docs/paper/main_vi.tex`.

**Tech Stack:** Python 3.13, pandas, numpy, scipy.stats (spearmanr), pytest.

## Global Constraints

- Read-only on existing artifacts; write NEW files only (`experiments/panel/panel_master.csv`, `experiments/panel/findings.md`). Do not modify panel.csv or any classified/contract file.
- Do NOT compile LaTeX; do NOT edit main_vi.tex. findings.md is markdown content.
- Do NOT retrain heavy models. If an artifact is missing, the driver degrades gracefully (skips that line) rather than crashing.
- Driver bootstraps sys.path with repo src/.
- Tests flat in tests/, pytest, import `from esgwash...`.

Verified input artifacts (all exist after Phases 0–3):
- `experiments/panel/panel.csv`: `bank, year, cti, nar, qdr, n_commit`.
- `experiments/panel/embedding_signals.csv`: `bank, year, bri`.
- `experiments/panel/say_do.csv`: `bank, year, pillar, cti_p, qdr_p, say_do, n`.
- `experiments/panel/figure_types.csv`: `bank, year, type, n`.
- `experiments/panel/summary.json`: descriptive + selective_disclosure + temporal_trend (from analyse_panel.py).
- `experiments/eval/transfer.csv`: `config, macro_f1, task`.

## File Structure

- Create `src/esgwash/indices/panel_master.py` — `merge_signals`, `rq4_correlations`, `say_do_by_pillar`.
- Create `experiments/build_findings.py` — driver: build master + write findings.md.
- Create `tests/test_panel_master.py`.

---

### Task 1: Master-panel merge + summary stats (pure)

**Files:**
- Create: `src/esgwash/indices/panel_master.py`
- Test: `tests/test_panel_master.py`

**Interfaces:**
- Produces:
  - `merge_signals(panel: pd.DataFrame, bri: pd.DataFrame, say_do: pd.DataFrame) -> pd.DataFrame` — left-merge on (bank,year); pivots say_do's `say_do` into columns `say_do_env, say_do_soc, say_do_gov`. Result columns: `bank, year, cti, nar, qdr, n_commit, bri, say_do_env, say_do_soc, say_do_gov` (missing pillars -> NaN).
  - `rq4_correlations(master: pd.DataFrame) -> dict` — Spearman of (cti, bri) and (nar, bri) over rows with both non-null; returns `{"cti_bri": {"rho":..,"p":..,"n":..}, "nar_bri": {...}}`.
  - `say_do_by_pillar(say_do: pd.DataFrame) -> dict` — mean `say_do` per pillar -> `{"env":..,"soc":..,"gov":..}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_panel_master.py
import numpy as np
import pandas as pd
from esgwash.indices import panel_master as pm


def _panel():
    return pd.DataFrame({"bank": ["a", "b"], "year": [2023, 2023],
                         "cti": [0.5, 0.2], "nar": [0.3, 0.4], "qdr": [0.2, 0.4],
                         "n_commit": [10, 20]})


def _bri():
    return pd.DataFrame({"bank": ["a", "b"], "year": [2023, 2023], "bri": [0.6, 0.5]})


def _say_do():
    rows = []
    for bank, sd in [("a", {"env": 0.1, "soc": -0.1, "gov": 0.3}),
                     ("b", {"env": 0.0, "soc": 0.2, "gov": 0.1})]:
        for p, v in sd.items():
            rows.append({"bank": bank, "year": 2023, "pillar": p,
                         "cti_p": 0.0, "qdr_p": 0.0, "say_do": v, "n": 5})
    return pd.DataFrame(rows)


def test_merge_signals_columns_and_values():
    m = pm.merge_signals(_panel(), _bri(), _say_do())
    assert set(["bank", "year", "cti", "nar", "qdr", "n_commit", "bri",
                "say_do_env", "say_do_soc", "say_do_gov"]).issubset(m.columns)
    a = m[m["bank"] == "a"].iloc[0]
    assert a["bri"] == 0.6 and abs(a["say_do_gov"] - 0.3) < 1e-9


def test_rq4_correlations_shape():
    m = pm.merge_signals(_panel(), _bri(), _say_do())
    out = pm.rq4_correlations(m)
    assert "cti_bri" in out and "nar_bri" in out
    assert out["cti_bri"]["n"] == 2 and "rho" in out["cti_bri"]


def test_say_do_by_pillar_means():
    out = pm.say_do_by_pillar(_say_do())
    assert abs(out["gov"] - 0.2) < 1e-9 and abs(out["env"] - 0.05) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_panel_master.py -v`
Expected: FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/esgwash/indices/panel_master.py
"""Merge all per-(bank,year) washing signals into one master panel + headline stats.

Signals: rubric CTI/NAR/QDR (panel.csv), embedding BRI (embedding_signals.csv),
per-pillar say-do gap (say_do.csv). Used to write the RQ1-RQ5 findings report.
"""
from __future__ import annotations

import pandas as pd
from scipy.stats import spearmanr

PILLARS = ("env", "soc", "gov")


def merge_signals(panel: pd.DataFrame, bri: pd.DataFrame, say_do: pd.DataFrame) -> pd.DataFrame:
    m = panel.merge(bri[["bank", "year", "bri"]], on=["bank", "year"], how="left")
    wide = say_do.pivot_table(index=["bank", "year"], columns="pillar",
                              values="say_do").reset_index()
    wide = wide.rename(columns={p: f"say_do_{p}" for p in PILLARS})
    for p in PILLARS:
        if f"say_do_{p}" not in wide.columns:
            wide[f"say_do_{p}"] = float("nan")
    keep = ["bank", "year"] + [f"say_do_{p}" for p in PILLARS]
    return m.merge(wide[keep], on=["bank", "year"], how="left")


def _spear(df: pd.DataFrame, a: str, b: str) -> dict:
    v = df.dropna(subset=[a, b])
    if len(v) < 3:
        return {"rho": float("nan"), "p": float("nan"), "n": int(len(v))}
    rho, p = spearmanr(v[a], v[b])
    return {"rho": round(float(rho), 4), "p": float(f"{p:.3e}"), "n": int(len(v))}


def rq4_correlations(master: pd.DataFrame) -> dict:
    return {"cti_bri": _spear(master, "cti", "bri"),
            "nar_bri": _spear(master, "nar", "bri")}


def say_do_by_pillar(say_do: pd.DataFrame) -> dict:
    g = say_do.groupby("pillar")["say_do"].mean()
    return {p: round(float(g.get(p, float("nan"))), 4) for p in PILLARS}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_panel_master.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/esgwash/indices/panel_master.py tests/test_panel_master.py
git commit -m "feat(indices): master-panel merge + RQ4/say-do summary stats"
```

---

### Task 2: Findings driver — panel_master.csv + findings.md

**Files:**
- Create: `experiments/build_findings.py`
- Test: `tests/test_panel_master.py` (append a driver smoke test)

**Interfaces:**
- Produces: `build_master(panel_dir="experiments/panel") -> pd.DataFrame` (loads panel/bri/say_do csvs, returns merged master; missing bri/say_do -> those columns NaN); `main(panel_dir="experiments/panel", eval_dir="experiments/eval") -> dict` — writes `panel_master.csv` + `findings.md`, returns a stats dict. Reads `summary.json` and `transfer.csv`/`figure_types.csv` if present (graceful skip otherwise).

- [ ] **Step 1: Write the failing test** (append to tests/test_panel_master.py)

```python
def test_build_master_smoke(tmp_path):
    import importlib.util, sys
    from pathlib import Path
    d = tmp_path
    _panel().to_csv(d / "panel.csv", index=False)
    _bri().to_csv(d / "embedding_signals.csv", index=False)
    _say_do().to_csv(d / "say_do.csv", index=False)
    spec = importlib.util.spec_from_file_location(
        "build_findings", Path(__file__).resolve().parents[1] / "experiments" / "build_findings.py")
    mod = importlib.util.module_from_spec(spec); sys.modules["build_findings"] = mod
    spec.loader.exec_module(mod)
    m = mod.build_master(panel_dir=str(d))
    assert "bri" in m.columns and "say_do_gov" in m.columns and len(m) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_panel_master.py::test_build_master_smoke -v`
Expected: FAIL (FileNotFoundError for experiments/build_findings.py).

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/build_findings.py
"""Phase 4: consolidate Phases 0-3 artifacts into panel_master.csv + findings.md
(paper-ready source for RQ1-RQ5). Read-only on inputs; no model training; no LaTeX."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from esgwash.indices.panel_master import merge_signals, rq4_correlations, say_do_by_pillar


def _read_csv(p: Path):
    return pd.read_csv(p) if p.exists() else None


def build_master(panel_dir: str = "experiments/panel") -> pd.DataFrame:
    d = Path(panel_dir)
    panel = pd.read_csv(d / "panel.csv")
    bri = _read_csv(d / "embedding_signals.csv")
    say_do = _read_csv(d / "say_do.csv")
    if bri is None:
        bri = panel[["bank", "year"]].assign(bri=float("nan"))
    if say_do is None:
        rows = [{"bank": b, "year": y, "pillar": p, "say_do": float("nan")}
                for b, y in panel[["bank", "year"]].itertuples(index=False) for p in ("env", "soc", "gov")]
        say_do = pd.DataFrame(rows)
    return merge_signals(panel, bri, say_do)


def main(panel_dir: str = "experiments/panel", eval_dir: str = "experiments/eval") -> dict:
    d = Path(panel_dir)
    master = build_master(panel_dir)
    master.to_csv(d / "panel_master.csv", index=False)

    say_do = _read_csv(d / "say_do.csv")
    summ = json.loads((d / "summary.json").read_text(encoding="utf-8")) if (d / "summary.json").exists() else {}
    transfer = _read_csv(Path(eval_dir) / "transfer.csv")
    figtypes = _read_csv(d / "figure_types.csv")

    rq4 = rq4_correlations(master)
    saydo = say_do_by_pillar(say_do) if say_do is not None else {}
    desc = summ.get("descriptive", {})

    L = ["# ESG-washing — consolidated findings (RQ1–RQ5)", "",
         f"Panel: {master['bank'].nunique()} banks x {master['year'].nunique()} years "
         f"= {len(master)} bank-years.", ""]
    L += ["## RQ1 — Prevalence"]
    for idx in ("cti", "nar", "qdr"):
        o = desc.get(f"{idx}_overall")
        if o:
            L.append(f"- {idx.upper()} = {o['mean']} (sd {o['sd']}, CI95 {o['ci95']})")
    L += ["", "## RQ2 — Selective disclosure"]
    sd2 = summ.get("selective_disclosure", {})
    if sd2:
        L.append(f"- pillar mean share: {sd2.get('pillar_mean_share')}")
        L.append(f"- Friedman chi2={sd2.get('friedman', {}).get('chi2')}, p={sd2.get('friedman', {}).get('p')}")
    L += ["", "## RQ3 — Temporal trend"]
    tr = summ.get("temporal_trend", {})
    for idx in ("cti", "qdr"):
        if idx in tr:
            L.append(f"- {idx.upper()}~year Spearman rho={tr[idx]['spearman_rho']} (p={tr[idx]['p']})")
    L += ["", "## RQ4 — Convergent validity (embedding BRI vs rubric)",
          f"- Spearman(CTI, BRI) = {rq4['cti_bri']['rho']} (p={rq4['cti_bri']['p']}, n={rq4['cti_bri']['n']})",
          f"- Spearman(NAR, BRI) = {rq4['nar_bri']['rho']} (p={rq4['nar_bri']['p']}, n={rq4['nar_bri']['n']})",
          "- Interpretation: boilerplate (cross-bank reused language) tracks NAMED actions, not vague "
          "cheap-talk; rubric CTI is not redundant with embedding similarity. (SBS dropped: anisotropy.)"]
    L += ["", "## RQ5 — Say-do gap + drivers"]
    if saydo:
        L.append(f"- mean say-do by pillar (CTI_p - QDR_p): {saydo} -> governance is the cheap-talk pillar.")
    if figtypes is not None:
        tot = figtypes.groupby("type")["n"].sum().sort_values(ascending=False)
        L.append(f"- quantified figures by type: {tot.to_dict()}")
    L += ["", "## Cross-lingual transfer (lexical baseline)"]
    if transfer is not None:
        for _, r in transfer.iterrows():
            L.append(f"- {r['task']} {r['config']}: macro-F1={r['macro_f1']}")
        L.append("- EN->VI collapse motivates translate-train / multilingual encoder.")

    (d / "findings.md").write_text("\n".join(L), encoding="utf-8")
    print(f"-> {d/'panel_master.csv'} ({len(master)} rows)")
    print(f"-> {d/'findings.md'}")
    print(f"RQ4 CTI~BRI rho={rq4['cti_bri']['rho']} p={rq4['cti_bri']['p']}; say-do {saydo}")
    return {"master": master, "rq4": rq4, "say_do": saydo}


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_panel_master.py -v`
Expected: PASS (4 passed total)

- [ ] **Step 5: Commit**

```bash
git add experiments/build_findings.py tests/test_panel_master.py
git commit -m "feat(indices): findings driver — panel_master.csv + RQ1-RQ5 findings.md"
```

---

## Self-Review

**Spec coverage (spec §7 Phase 4 — panel integration + RQ4/RQ5 writeup):**
- Merge CTI/NAR/QDR + BRI + say-do into one panel → Task 1 `merge_signals` + Task 2 `build_master`. ✓
- RQ4/RQ5 writeup with real numbers → Task 2 `findings.md` (RQ1–RQ5 + transfer). ✓
- Paper-ready source (no LaTeX compile, user folds into main_vi.tex) → scope note + findings.md. ✓
- Graceful degradation if an artifact missing → Task 2 `_read_csv` guards. ✓

**Placeholder scan:** none.

**Type consistency:** `merge_signals->DataFrame` consumed by `rq4_correlations`/`build_master`/driver; `rq4_correlations->dict[cti_bri,nar_bri]` and `say_do_by_pillar->dict[env,soc,gov]` consumed by `main`. ✓

**Note for executor:** Task 2 appends a test to tests/test_panel_master.py from Task 1 (reuses `_panel`/`_bri`/`_say_do`).
