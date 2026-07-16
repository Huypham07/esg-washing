import pandas as pd
from esgwash.models import spec_features as sf


def _clf():
    return pd.DataFrame({
        "bank": ["a", "a"], "year": [2023, 2023], "doc_id": ["a_2023", "a_2023"],
        "chunk_index": [0, 4], "content_text": ["green pledge", "giảm 30% phát thải 2030"],
        "token_count": [3, 6], "char_count": [11, 24],
        "is_env": [1, 1], "is_soc": [0, 0], "is_gov": [0, 0],
        "p_env": [0.8, 0.9], "p_soc": [0.1, 0.1], "p_gov": [0.1, 0.1],
        "is_commitment": [1, 1], "p_commitment": [0.7, 0.8], "spec_level": [0, 2]})


def test_feature_columns_have_no_identifiers():
    X, y = sf.chunk_features(_clf())
    for banned in ["bank", "year", "doc_id", "ticker", "chunk_index"]:
        assert banned not in X.columns
    assert list(y) == [0, 2]


def test_digit_and_year_features():
    X, y = sf.chunk_features(_clf())
    r1 = X.iloc[1]   # "giảm 30% phát thải 2030"
    assert r1["has_digit"] == 1 and r1["has_year"] == 1
    assert r1["n_digit_runs"] >= 2
    r0 = X.iloc[0]   # "green pledge"
    assert r0["has_digit"] == 0 and r0["has_year"] == 0


def test_rel_position():
    X, y = sf.chunk_features(_clf())
    # doc max chunk_index = 4 -> positions 0/4=0.0 and 4/4=1.0
    assert X.iloc[0]["rel_position"] == 0.0
    assert X.iloc[1]["rel_position"] == 1.0


def test_train_eval_runs_on_toy():
    import importlib.util, sys, numpy as np
    from pathlib import Path
    import pandas as pd
    spec = importlib.util.spec_from_file_location(
        "saydo_shap", Path(__file__).resolve().parents[1] / "experiments" / "saydo_shap.py")
    mod = importlib.util.module_from_spec(spec); sys.modules["saydo_shap"] = mod
    spec.loader.exec_module(mod)
    rng = np.random.default_rng(0)
    n = 60
    X = pd.DataFrame({c: rng.random(n) for c in __import__(
        "esgwash.models.spec_features", fromlist=["FEATURE_COLS"]).FEATURE_COLS})
    y = pd.Series(([0, 1, 2] * (n // 3)))
    out = mod.train_eval(X, y)
    assert "macro_f1_cv" in out and 0.0 <= out["macro_f1_cv"] <= 1.0
    assert out["n"] == n
