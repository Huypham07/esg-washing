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
    assert out[0] > 0.99            # [1,0] aligns with [1,0.01]
    assert abs(out[1]) < 0.2        # [0,1] orthogonal to both


def test_pairwise_max_cosine_empty_B():
    A = _unit([[1, 0]])
    out = al.pairwise_max_cosine(A, np.empty((0, 2)))
    assert out.shape == (1,) and out[0] == 0.0


def test_sbs_vague_far_from_quantified_is_low():
    # vague chunk orthogonal to the single quantified chunk -> backing ~0
    emb = _unit([[1, 0], [0, 1]])
    spec = np.array([0, 2])
    sbs, per = al.substance_backing_score(emb, spec)
    assert per.shape == (1,)
    assert sbs < 0.2


def test_sbs_vague_near_quantified_is_high():
    emb = _unit([[1, 0.02], [1, 0]])
    spec = np.array([0, 2])
    sbs, _ = al.substance_backing_score(emb, spec)
    assert sbs > 0.99


def test_sbs_no_quantified_gives_zero_backing():
    emb = _unit([[1, 0], [0, 1]])
    spec = np.array([0, 0])
    sbs, per = al.substance_backing_score(emb, spec)
    assert sbs == 0.0 and list(per) == [0.0, 0.0]


def test_sbs_no_vague_is_nan():
    emb = _unit([[1, 0]])
    spec = np.array([2])
    sbs, per = al.substance_backing_score(emb, spec)
    assert np.isnan(sbs) and per.size == 0


def test_bri_identical_other_bank_text_is_high():
    emb = _unit([[1, 0], [1, 0.01], [0, 1]])
    banks = np.array(["a", "b", "a"])
    bri, per = al.boilerplate_reuse_index(emb, banks)
    assert per.shape == (3,)
    assert per[0] > 0.99            # a's [1,0] matches b's [1,0.01]
    assert 0.0 <= bri <= 1.0001


def test_bri_single_bank_is_nan():
    emb = _unit([[1, 0], [0, 1]])
    banks = np.array(["a", "a"])
    bri, per = al.boilerplate_reuse_index(emb, banks)
    assert np.isnan(bri)


def test_compute_with_fake_embedder():
    import importlib.util, sys
    from pathlib import Path
    import pandas as pd
    spec = importlib.util.spec_from_file_location(
        "embedding_signals", Path(__file__).resolve().parents[1] / "experiments" / "embedding_signals.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["embedding_signals"] = mod
    spec.loader.exec_module(mod)

    # 2 banks, 1 year; bank a: vague + quantified aligned; bank b: vague alone
    clf = pd.DataFrame({
        "bank": ["a", "a", "b"], "year": [2023, 2023, 2023],
        "content_text": ["green pledge", "green 5000bn", "vague aspiration"],
        "is_env": [1, 1, 1], "is_soc": [0, 0, 0], "is_gov": [0, 0, 0],
        "is_commitment": [1, 1, 1], "spec_level": [0, 2, 0],
    })

    class FakeEmb:
        def embed(self, texts):
            import numpy as np
            m = {"green pledge": [1, 0], "green 5000bn": [1, 0.01],
                 "vague aspiration": [0, 1]}
            v = np.array([m[t] for t in texts], dtype=float)
            return v / np.linalg.norm(v, axis=1, keepdims=True)

    out = mod.compute(clf, FakeEmb())
    a = out[out["bank"] == "a"].iloc[0]
    b = out[out["bank"] == "b"].iloc[0]
    assert a["sbs"] > 0.99          # a's vague backed by its quantified
    assert b["sbs"] == 0.0          # b has no quantified chunk
