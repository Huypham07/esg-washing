import pandas as pd
from esgwash.eda import corpus_eda as ce


def _toy_clf():
    return pd.DataFrame({
        "bank": ["a", "a", "a", "b"], "year": [2023, 2023, 2023, 2023],
        "chunk_id": ["c0", "c1", "c2", "c3"],
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
    assert comm.loc["a", 2023] == 2
