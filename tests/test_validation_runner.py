import pandas as pd
from esgwash.validation.runner import sample_for_audit, audit_agreement


def test_sample_balanced_per_pillar():
    df = pd.DataFrame({
        "bank": ["b"] * 6, "year": [2023] * 6,
        "pillar": ["env", "env", "soc", "soc", "gov", "gov"],
        "chunk_index": range(6), "content_text": [f"t{i}" for i in range(6)],
        "spec_level": [0, 1, 2, 0, 1, 2],
    })
    s = sample_for_audit(df, n_per_pillar=1, seed=0)
    assert set(s["pillar"]) == {"env", "soc", "gov"} and len(s) == 3
    assert "gold_level" in s.columns      # cot trong de gan tay


def test_agreement_simple():
    audited = pd.DataFrame({"spec_level": [0, 1, 2, 2], "gold_level": [0, 1, 2, 1]})
    acc = audit_agreement(audited)
    assert abs(acc - 0.75) < 1e-9
