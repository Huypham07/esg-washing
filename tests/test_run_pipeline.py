"""Unit test gate denominator (to_long): chi cam ket co gan tru moi vao CTI."""
import pandas as pd

from esgwash.run import to_long


def _classified():
    return pd.DataFrame({
        "doc_id": ["d", "d", "d"], "chunk_index": [0, 1, 2],
        "bank": ["b"] * 3, "year": [2023] * 3,
        "content_text": ["t0", "t1", "t2"],
        "is_env": [1, 0, 1], "is_soc": [0, 0, 0], "is_gov": [0, 1, 0],
        "is_commitment": [1, 1, 0], "spec_level": [0, 1, 0],
    })


def test_to_long_keeps_only_pillar_positive_rows():
    long = to_long(_classified())
    # chunk0->env, chunk1->gov, chunk2->env ; chunk khong tru bi loai
    assert set(zip(long["chunk_index"], long["pillar"])) == {(0, "env"), (1, "gov"), (2, "env")}


def test_to_long_carries_spec_level_for_index():
    long = to_long(_classified())
    env = long[(long["pillar"] == "env") & (long["chunk_index"] == 0)]
    assert env["spec_level"].iloc[0] == 0
