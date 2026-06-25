import pandas as pd
import pytest


def test_atomic_iaa_perfect_and_disagree():
    from experiments.iaa_atomic import atomic_iaa

    a = pd.DataFrame({
        "chunk_id": [1, 2, 3, 4],
        "co_cam_ket": [1, 1, 0, 0],
        "g_soc": [1, 0, 1, 0],
    })
    b = pd.DataFrame({
        "chunk_id": [1, 2, 3, 4],
        "co_cam_ket": [1, 1, 0, 0],
        "g_soc": [0, 0, 1, 0],
    })
    r = atomic_iaa(a, b).set_index("field")

    assert r.loc["co_cam_ket", "kappa"] == 1.0
    assert r.loc["co_cam_ket", "n_disagree"] == 0
    assert r.loc["g_soc", "n_disagree"] == 1


def test_atomic_iaa_constant_side_returns_nan():
    """If one annotator is constant for a field, kappa should be NaN."""
    import math
    from experiments.iaa_atomic import atomic_iaa

    a = pd.DataFrame({
        "chunk_id": [1, 2, 3],
        "co_cam_ket": [0, 0, 0],  # constant
    })
    b = pd.DataFrame({
        "chunk_id": [1, 2, 3],
        "co_cam_ket": [1, 0, 1],
    })
    r = atomic_iaa(a, b).set_index("field")
    assert math.isnan(r.loc["co_cam_ket", "kappa"])


def test_atomic_iaa_drops_nan_rows():
    """Rows where either annotator has NaN for a field should be dropped."""
    from experiments.iaa_atomic import atomic_iaa

    a = pd.DataFrame({
        "chunk_id": [1, 2, 3],
        "co_cam_ket": [1.0, float("nan"), 0.0],
    })
    b = pd.DataFrame({
        "chunk_id": [1, 2, 3],
        "co_cam_ket": [1.0, 0.0, 0.0],
    })
    r = atomic_iaa(a, b).set_index("field")
    # chunk 2 dropped; n=2
    assert r.loc["co_cam_ket", "n"] == 2
