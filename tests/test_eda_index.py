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
