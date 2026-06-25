import math

import pandas as pd
import pytest


def test_atomic_iaa_perfect_and_disagree():
    from experiments.iaa_atomic import atomic_iaa

    a = pd.DataFrame({
        "chunk_id": [1, 2, 3, 4],
        "co_hanh_dong_ten": [1, 1, 0, 0],  # was co_cam_ket; removed from FIELDS
        "g_soc": [1, 0, 1, 0],
    })
    b = pd.DataFrame({
        "chunk_id": [1, 2, 3, 4],
        "co_hanh_dong_ten": [1, 1, 0, 0],
        "g_soc": [0, 0, 1, 0],
    })
    r = atomic_iaa(a, b).set_index("field")

    assert r.loc["co_hanh_dong_ten", "kappa"] == 1.0
    assert r.loc["co_hanh_dong_ten", "n_disagree"] == 0
    assert r.loc["g_soc", "n_disagree"] == 1
    # co_cam_ket must not appear in results (removed from FIELDS)
    assert "co_cam_ket" not in r.index


def test_atomic_iaa_constant_side_returns_nan():
    """If one annotator is constant for a field, kappa should be NaN."""
    from experiments.iaa_atomic import atomic_iaa

    a = pd.DataFrame({
        "chunk_id": [1, 2, 3],
        "co_hanh_dong_ten": [0, 0, 0],  # constant
    })
    b = pd.DataFrame({
        "chunk_id": [1, 2, 3],
        "co_hanh_dong_ten": [1, 0, 1],
    })
    r = atomic_iaa(a, b).set_index("field")
    assert math.isnan(r.loc["co_hanh_dong_ten", "kappa"])


def test_atomic_iaa_drops_nan_rows():
    """Rows where either annotator has NaN for a field should be dropped."""
    from experiments.iaa_atomic import atomic_iaa

    a = pd.DataFrame({
        "chunk_id": [1, 2, 3],
        "co_hanh_dong_ten": [1.0, float("nan"), 0.0],
    })
    b = pd.DataFrame({
        "chunk_id": [1, 2, 3],
        "co_hanh_dong_ten": [1.0, 0.0, 0.0],
    })
    r = atomic_iaa(a, b).set_index("field")
    # chunk 2 dropped; n=2
    assert r.loc["co_hanh_dong_ten", "n"] == 2
