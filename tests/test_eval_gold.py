"""Structural test for experiments/eval_gold.py — no GPU required.

Checks that BIN contains the 5 atomic flag names introduced in Task 6.
Importing eval_gold at module level must NOT trigger model loading
(load_models is called only inside main()).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# experiments/ is not a package; add it to sys.path so the import works.
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))

import experiments.eval_gold as eg  # noqa: E402
from esgwash.models.specificity_llm import derive_flags  # noqa: E402


def test_eval_gold_has_atomic_bins():
    """BIN must contain all 5 atomic flag entries."""
    names = {b[0] for b in eg.BIN}
    assert {"co_cam_ket", "co_hanh_dong_ten", "co_so_dinh_luong", "quy_ve_bank", "co_moc_tg"} <= names


# ── Fix 1: derive_spec_level_from_row helper ─────────────────────────────────

def test_derive_spec_level_from_row_committed():
    """For committed rows, helper must match derive_flags rule."""
    # Muc 2: quant + bank
    row = {"co_cam_ket": 1, "co_so_dinh_luong": 1, "quy_ve_bank": 1,
           "co_hanh_dong_ten": 0, "co_moc_tg": 0}
    assert eg._derive_spec_level_from_row(row) == 2

    # Muc 1: action only
    row2 = {"co_cam_ket": 1, "co_so_dinh_luong": 0, "quy_ve_bank": 0,
            "co_hanh_dong_ten": 1, "co_moc_tg": 0}
    assert eg._derive_spec_level_from_row(row2) == 1

    # Muc 0: nothing
    row3 = {"co_cam_ket": 1, "co_so_dinh_luong": 0, "quy_ve_bank": 0,
            "co_hanh_dong_ten": 0, "co_moc_tg": 0}
    assert eg._derive_spec_level_from_row(row3) == 0


def test_derive_spec_level_from_row_not_committed():
    """For non-committed rows, helper must return NaN."""
    row = {"co_cam_ket": 0, "co_so_dinh_luong": 1, "quy_ve_bank": 1,
           "co_hanh_dong_ten": 1, "co_moc_tg": 0}
    result = eg._derive_spec_level_from_row(row)
    assert result is np.nan or (isinstance(result, float) and np.isnan(result))


def test_derive_spec_level_matches_derive_flags():
    """Helper output must equal derive_flags(...)[ 1] for committed rows."""
    flags_in = {"co_so_dinh_luong": 1, "quy_ve_bank": 1, "co_hanh_dong_ten": 0}
    row = {"co_cam_ket": 1, **flags_in, "co_moc_tg": 0}
    _, expected_level = derive_flags(flags_in)
    assert eg._derive_spec_level_from_row(row) == expected_level


def test_load_gold_spec_level_is_rule_derived():
    """load_gold() must produce g_spec_level_A/B derived purely from atomic flags.

    For committed rows: derived value must match derive_flags on the same flags.
    This is a data-integrity check (GPU-free, pandas only).
    """
    gold_a = ROOT / "data/gold_annot_1_relabeled.xlsx"
    gold_b = ROOT / "data/gold_annot_2_relabeled.xlsx"
    if not gold_a.exists() or not gold_b.exists():
        import pytest
        pytest.skip("Gold xlsx files not found — skipping I/O-dependent test")

    m = eg.load_gold()

    for side in ("A", "B"):
        committed_mask = m[f"co_cam_ket_{side}"] == 1
        committed = m[committed_mask]
        for _, row in committed.iterrows():
            flags_in = {
                "co_so_dinh_luong": row[f"co_so_dinh_luong_{side}"],
                "quy_ve_bank": row[f"quy_ve_bank_{side}"],
                "co_hanh_dong_ten": row[f"co_hanh_dong_ten_{side}"],
            }
            _, expected = derive_flags(flags_in)
            derived = row[f"g_spec_level_{side}"]
            assert derived == expected, (
                f"chunk_id={row['chunk_id']}, side={side}: "
                f"g_spec_level_{side}={derived} != rule-derived {expected}"
            )


# ── Fix 4: _spec_scores QWK symmetry ─────────────────────────────────────────

def test_spec_scores_nan_when_yp_constant():
    """QWK must be nan when yp is constant (symmetric with yt-constant guard)."""
    yt = np.array([0, 1, 2, 0, 1])
    yp = np.array([1, 1, 1, 1, 1])   # constant yp
    result = eg._spec_scores(yt, yp)
    assert np.isnan(result["quadratic_weighted_kappa"]), (
        "QWK must be nan when yp is constant"
    )


def test_spec_scores_nan_when_yt_constant():
    """QWK must be nan when yt is constant (existing guard, confirm still works)."""
    yt = np.array([2, 2, 2])
    yp = np.array([0, 1, 2])
    result = eg._spec_scores(yt, yp)
    assert np.isnan(result["quadratic_weighted_kappa"])
