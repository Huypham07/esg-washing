"""Unit test CTI/NAR/QDR tu spec_level tren du lieu nho tu tao."""
import pandas as pd

from esgwash.indices.cti import compute_specificity_shares, build_index_table


def _toy():
    # 4 commitment ESG (is_env=1), spec_level=[0,0,1,2] -> CTI=.5 NAR=.25 QDR=.25
    return pd.DataFrame({
        "bank": ["b"] * 4, "year": [2023] * 4,
        "is_env": [1, 1, 1, 1], "is_soc": [0, 0, 0, 0], "is_gov": [0, 0, 0, 0],
        "is_commitment": [1, 1, 1, 1], "spec_level": [0, 0, 1, 2],
    })


def _toy_with_nonesg():
    # 4 commitment ESG + 2 commitment non-ESG (is_env/soc/gov=0) -> non-ESG excluded
    base = _toy()
    extra = pd.DataFrame({
        "bank": ["b"] * 2, "year": [2023] * 2,
        "is_env": [0, 0], "is_soc": [0, 0], "is_gov": [0, 0],
        "is_commitment": [1, 1], "spec_level": [2, 2],
    })
    return pd.concat([base, extra], ignore_index=True)


def test_shares_sum_to_one_and_match_counts():
    out = compute_specificity_shares(_toy(), n_resamples=200)
    row = out.iloc[0]
    assert row["n_commit"] == 4
    assert row["cti"] == 0.5 and row["nar"] == 0.25 and row["qdr"] == 0.25
    assert abs(row["cti"] + row["nar"] + row["qdr"] - 1.0) < 1e-9


def test_ci_brackets_point():
    out = compute_specificity_shares(_toy(), n_resamples=200)
    row = out.iloc[0]
    assert row["cti_lo"] <= row["cti"] <= row["cti_hi"]


def test_non_esg_commitment_excluded_from_denominator():
    out = compute_specificity_shares(_toy_with_nonesg(), n_resamples=200)
    row = out.iloc[0]
    assert row["n_commit"] == 4  # 2 non-ESG bi loai


def test_build_index_table_has_all_three():
    out = build_index_table(_toy(), n_resamples=200)
    for c in ["cti", "nar", "qdr", "cti_lo", "qdr_hi", "n_commit"]:
        assert c in out.columns
    assert "pillar" not in out.columns
