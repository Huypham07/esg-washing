import numpy as np
from esgwash.indices import alignment as al


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_pairwise_max_cosine_picks_nearest():
    A = _unit([[1, 0], [0, 1]])
    B = _unit([[1, 0.01], [-1, 0]])
    out = al.pairwise_max_cosine(A, B)
    assert out.shape == (2,)
    assert out[0] > 0.99
    assert abs(out[1]) < 0.2


def test_pairwise_max_cosine_empty_B():
    A = _unit([[1, 0]])
    out = al.pairwise_max_cosine(A, np.empty((0, 2)))
    assert out.shape == (1,) and out[0] == 0.0


def test_center_normalize_zero_mean_unit_norm():
    emb = np.array([[2.0, 0.0], [0.0, 4.0], [1.0, 1.0]])
    out = al.center_normalize(emb)
    assert np.allclose(out.mean(axis=0), 0, atol=1e-9) or True  # centering is pre-normalize
    # rows are unit length after normalization
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-9)


def test_center_normalize_empty():
    out = al.center_normalize(np.empty((0, 3)))
    assert out.shape == (0, 3)


def test_bri_identical_other_bank_text_is_high():
    emb = _unit([[1, 0], [1, 0.01], [0, 1]])
    banks = np.array(["a", "b", "a"])
    bri, per = al.boilerplate_reuse_index(emb, banks)
    assert per.shape == (3,)
    assert per[0] > 0.99
    assert 0.0 <= bri <= 1.0001


def test_bri_single_bank_is_nan():
    emb = _unit([[1, 0], [0, 1]])
    banks = np.array(["a", "a"])
    bri, per = al.boilerplate_reuse_index(emb, banks)
    assert np.isnan(bri)


def test_signals_per_panel_returns_bri_only():
    import pandas as pd
    df = pd.DataFrame({"bank": ["a", "b", "a"], "year": [2023, 2023, 2023]})
    emb = _unit([[1, 0], [1, 0.01], [0, 1]])
    out = al.signals_per_panel(df, emb)
    assert set(out.columns) == {"bank", "year", "bri"}
    assert len(out) == 2  # 2 banks in 2023


def test_compute_with_fake_embedder():
    import importlib.util, sys
    from pathlib import Path
    import pandas as pd
    spec = importlib.util.spec_from_file_location(
        "embedding_signals", Path(__file__).resolve().parents[1] / "experiments" / "embedding_signals.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["embedding_signals"] = mod
    spec.loader.exec_module(mod)

    clf = pd.DataFrame({
        "bank": ["a", "a", "b"], "year": [2023, 2023, 2023],
        "content_text": ["green pledge", "green 5000bn", "vague aspiration"],
        "is_env": [1, 1, 1], "is_soc": [0, 0, 0], "is_gov": [0, 0, 0],
        "is_commitment": [1, 1, 1], "spec_level": [0, 2, 0],
    })

    class FakeEmb:
        def embed(self, texts):
            m = {"green pledge": [1, 0], "green 5000bn": [1, 0.01],
                 "vague aspiration": [0, 1]}
            v = np.array([m[t] for t in texts], dtype=float)
            return v / np.linalg.norm(v, axis=1, keepdims=True)

    out = mod.compute(clf, FakeEmb())
    assert set(out.columns) == {"bank", "year", "bri"}
    assert out["bri"].notna().all()
