# Final-Fix Report — attribute-extraction core code review

**Date:** 2026-06-25  
**Branch:** washing-depth  
**Commit:** (see SHA below)

---

## Fix 1 (Important) — derive g_spec_level via shared rule

**What changed:**
- Added `_derive_spec_level_from_row(row)` helper in `experiments/eval_gold.py` — pure function, calls `derive_flags()` from `esgwash.models.specificity_llm`, returns `float('nan')` for non-committed rows.
- Imported `derive_flags` at top of `eval_gold.py`.
- `load_gold()` now calls this helper for both `_A` and `_B` sides and overwrites `g_spec_level_A`/`g_spec_level_B` with the derived values.
- Consistency check (warning print) fires if any pre-baked non-null value differs from derived — warns with chunk_ids and count.

**TDD evidence:**  
Tests written first (failing), then implementation:
- `test_derive_spec_level_from_row_committed` — Muc 0/1/2 cases
- `test_derive_spec_level_from_row_not_committed` — returns NaN for co_cam_ket==0
- `test_derive_spec_level_matches_derive_flags` — helper == derive_flags(...)
- `test_load_gold_spec_level_is_rule_derived` — reads real xlsx, asserts for all committed rows both sides

**derived==prebaked assert result:**  
`test_load_gold_spec_level_is_rule_derived` PASSED on real `data/gold_annot_*_relabeled.xlsx` — 0 mismatches across all committed rows on both annotator sides. No warnings printed.

---

## Fix 2 (Important, doc) — BIN comment on commit confounding

**What changed:**  
Added 3-line comment above `BIN` in `experiments/eval_gold.py` noting: the `commit` row uses topic-gated `is_commitment` (confounded by topic-head errors); the `co_cam_ket` row is the unconfounded commitment-intent comparison.

**Tests:** No new test needed — doc-only change.

---

## Fix 3 (Minor) — remove dead `import os` in iaa_atomic.py

**What changed:**  
Grepped `iaa_atomic.py` — `os` appears only on the `import os` line, never used elsewhere. Removed the dead import.

**Tests:** No test needed — dead-import removal.

---

## Fix 4 (Minor) — _spec_scores QWK guard symmetry

**What changed:**  
`experiments/eval_gold.py` `_spec_scores`: changed guard from `len(set(yt)) > 1` to `len(set(yt)) > 1 and len(set(yp)) > 1`, matching the symmetry of `_binary_scores`. QWK now returns nan when either `yt` or `yp` is constant.

**TDD evidence:**
- `test_spec_scores_nan_when_yp_constant` — was FAILING before fix, PASSING after
- `test_spec_scores_nan_when_yt_constant` — was already passing, still passes

---

## Test output

```
20 passed in 3.20s
tests/test_eval_gold.py::test_eval_gold_has_atomic_bins PASSED
tests/test_eval_gold.py::test_derive_spec_level_from_row_committed PASSED
tests/test_eval_gold.py::test_derive_spec_level_from_row_not_committed PASSED
tests/test_eval_gold.py::test_derive_spec_level_matches_derive_flags PASSED
tests/test_eval_gold.py::test_load_gold_spec_level_is_rule_derived PASSED
tests/test_eval_gold.py::test_spec_scores_nan_when_yp_constant PASSED
tests/test_eval_gold.py::test_spec_scores_nan_when_yt_constant PASSED
tests/test_specificity_llm.py: 13 tests PASSED (all pre-existing)
```

---

## Concerns

None. All 4 fixes applied cleanly. The real-data assert passing confirms the pre-baked xlsx g_spec_level values were already rule-consistent — no data drift.
