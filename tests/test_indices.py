"""Unit test CTI/NAR/QDR tu spec_level tren du lieu nho tu tao."""
import pandas as pd

from esgwash.indices.cti import compute_specificity_shares, build_index_table


def _toy():
    # 1 o (b,y,env): 4 commitment, spec_level = [0,0,1,2] -> CTI .5 NAR .25 QDR .25
    return pd.DataFrame({
        "bank": ["b"] * 4, "year": [2023] * 4, "pillar": ["env"] * 4,
        "is_commitment": [1, 1, 1, 1], "spec_level": [0, 0, 1, 2],
    })


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


def test_build_index_table_has_all_three():
    out = build_index_table(_toy(), n_resamples=200)
    for c in ["cti", "nar", "qdr", "cti_lo", "qdr_hi", "n_commit"]:
        assert c in out.columns
