import numpy as np
import pandas as pd
from esgwash.models import interpret as it


def _toy():
    pos = ["green energy target reduce emissions"] * 20
    neg = ["the bank board meeting quarterly report"] * 20
    df = pd.DataFrame({
        "text": pos + neg,
        "text_en": pos + neg,
        "env": [1] * 20 + [0] * 20,
    })
    return df


def test_top_tokens_returns_sorted_k():
    toks = it.top_tokens(_toy(), "env", text_col="text", k=5)
    assert len(toks) == 5
    weights = [w for _, w in toks]
    assert weights == sorted(weights, reverse=True)
    # an env-positive word should surface
    assert any(t in {"green", "energy", "emissions", "reduce", "target"}
               for t, _ in toks)


def test_cross_lingual_macro_f1_in_range():
    df = _toy()
    f1 = it.cross_lingual_macro_f1(df, df, ["env"], "text", "text", seed=0)
    assert 0.0 <= f1 <= 1.0
    assert f1 > 0.8   # toy is linearly separable


def test_fit_head_skips_nan_rows():
    df = _toy()
    df.loc[0, "env"] = np.nan       # one NaN label must be dropped, no crash
    pipe = it.fit_head(df, "env", "text")
    assert pipe.predict(["green energy emissions"]).shape == (1,)
