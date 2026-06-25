"""Unit test gate denominator (to_long): chi cam ket co gan tru moi vao CTI."""
import pandas as pd

from esgwash.run import classify_chunks, to_long

# ---------------------------------------------------------------------------
# Shared stubs
# ---------------------------------------------------------------------------

FLAG_COLS = ["co_cam_ket", "co_hanh_dong_ten", "co_so_dinh_luong", "quy_ve_bank", "co_moc_tg"]


class _StubTopic:
    def predict(self, texts):
        n = len(texts)
        return pd.DataFrame({"env": [0.9] * n, "soc": [0.1] * n, "gov": [0.1] * n,
                             "is_env": [1] * n, "is_soc": [0] * n, "is_gov": [0] * n,
                             "pillar": ["env"] * n})


class _StubCommit:
    def predict(self, texts):
        return pd.DataFrame({"p_commitment": [0.9] * len(texts),
                             "is_commitment": [1] * len(texts)})


class _StubSpec:
    """Returns all 5 atomic flags + evidence (co_cam_ket=0 -> gate will block)."""
    def predict(self, texts):
        n = len(texts)
        return pd.DataFrame({
            "p_specificity": [0.0] * n, "spec_level": [0] * n,
            "is_specific": [0] * n, "parse_ok": [True] * n,
            "rubric": ["{}"] * n, "raw": ["{}"] * n, "evidence": ["{}"] * n,
            "co_cam_ket": [0] * n, "co_hanh_dong_ten": [0] * n,
            "co_so_dinh_luong": [0] * n, "quy_ve_bank": [0] * n,
            "co_moc_tg": [0] * n,
        })


# ---------------------------------------------------------------------------
# to_long tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# classify_chunks: 5 atomic flags + evidence propagation
# ---------------------------------------------------------------------------


def test_classify_commit_gate_uses_flags():
    """Task 4: 5 atomic flag cols + evidence must appear in output."""
    chunks = pd.DataFrame({"content_text": ["x"], "doc_id": ["d"], "chunk_index": [0]})
    out = classify_chunks(chunks, _StubTopic(), _StubCommit(), _StubSpec())
    assert "co_so_dinh_luong" in out.columns
    for col in FLAG_COLS:
        assert col in out.columns, f"Missing flag column: {col}"
    assert "evidence" in out.columns


def test_classify_flag_columns_present_on_non_commit_rows():
    """Non-commit rows (is_commitment=0 from PhoBERT) must have flag cols filled 0, not NaN."""

    class StubCommitNo:
        def predict(self, texts):
            return pd.DataFrame({"p_commitment": [0.1] * len(texts),
                                 "is_commitment": [0] * len(texts)})

    chunks = pd.DataFrame({"content_text": ["x"], "doc_id": ["d"], "chunk_index": [0]})
    out = classify_chunks(chunks, _StubTopic(), StubCommitNo(), _StubSpec())
    for col in FLAG_COLS:
        assert col in out.columns, f"Missing flag column: {col}"
        assert out[col].iloc[0] == 0, f"Non-commit row should have {col}=0"
    assert "evidence" in out.columns
    assert out["evidence"].iloc[0] == ""


def test_classify_final_is_commitment_gate():
    """Final is_commitment = co_cam_ket AND (is_env OR is_soc OR is_gov).
    PhoBERT says is_commitment=1, but co_cam_ket=0 -> final must be 0."""
    chunks = pd.DataFrame({"content_text": ["x"], "doc_id": ["d"], "chunk_index": [0]})
    # _StubSpec returns co_cam_ket=0, _StubTopic returns is_env=1
    out = classify_chunks(chunks, _StubTopic(), _StubCommit(), _StubSpec())
    # co_cam_ket=0 so final is_commitment must be 0
    assert out["is_commitment"].iloc[0] == 0


def test_classify_final_is_commitment_passes_when_flag_set():
    """co_cam_ket=1 AND is_env=1 -> final is_commitment=1."""

    class StubSpecCommit:
        def predict(self, texts):
            n = len(texts)
            return pd.DataFrame({
                "p_specificity": [0.8] * n, "spec_level": [1] * n,
                "is_specific": [1] * n, "parse_ok": [True] * n,
                "rubric": ["{}"] * n, "raw": ["{}"] * n, "evidence": ["some"] * n,
                "co_cam_ket": [1] * n, "co_hanh_dong_ten": [1] * n,
                "co_so_dinh_luong": [0] * n, "quy_ve_bank": [0] * n,
                "co_moc_tg": [0] * n,
            })

    chunks = pd.DataFrame({"content_text": ["x"], "doc_id": ["d"], "chunk_index": [0]})
    out = classify_chunks(chunks, _StubTopic(), _StubCommit(), StubSpecCommit())
    # co_cam_ket=1 AND is_env=1 -> final is_commitment=1
    assert out["is_commitment"].iloc[0] == 1


def test_classify_only_scores_commitments():
    """spec_level remains 0 for non-commit rows (spec_on_commitment=True, default)."""

    class StubCommitMixed:
        def predict(self, texts):
            n = len(texts)
            vals = [1 if i == 0 else 0 for i in range(n)]
            return pd.DataFrame({"p_commitment": [0.9 if v else 0.1 for v in vals],
                                 "is_commitment": vals})

    class StubSpecLevel1:
        def predict(self, texts):
            n = len(texts)
            return pd.DataFrame({
                "p_specificity": [0.9] * n, "spec_level": [1] * n,
                "is_specific": [1] * n, "parse_ok": [True] * n,
                "rubric": ["{}"] * n, "raw": ["{}"] * n, "evidence": ["ev"] * n,
                "co_cam_ket": [1] * n, "co_hanh_dong_ten": [1] * n,
                "co_so_dinh_luong": [0] * n, "quy_ve_bank": [0] * n,
                "co_moc_tg": [0] * n,
            })

    chunks = pd.DataFrame({"content_text": ["commit_row", "non_commit_row"],
                           "doc_id": ["d", "d"], "chunk_index": [0, 1]})
    out = classify_chunks(chunks, _StubTopic(), StubCommitMixed(), StubSpecLevel1())
    # non-commit row (index 1) must have spec_level=0 (not scored by LLM)
    assert out.loc[1, "spec_level"] == 0


def test_classify_spec_on_commitment_false_all_rows_get_llm_flags():
    """Regression guard for experiments/eval_gold.py: spec_on_commitment=False scores
    ALL rows via the LLM, even those the PhoBERT pre-filter marks is_commitment=0.
    The atomic flags must carry the LLM value, not the initialised 0."""

    class StubCommitNo:
        def predict(self, texts):
            return pd.DataFrame({"p_commitment": [0.1] * len(texts),
                                 "is_commitment": [0] * len(texts)})

    class StubSpecCommit:
        def predict(self, texts):
            n = len(texts)
            return pd.DataFrame({
                "p_specificity": [0.8] * n, "spec_level": [1] * n,
                "is_specific": [1] * n, "parse_ok": [True] * n,
                "rubric": ["{}"] * n, "raw": ["{}"] * n, "evidence": ["ev"] * n,
                "co_cam_ket": [1] * n, "co_hanh_dong_ten": [0] * n,
                "co_so_dinh_luong": [0] * n, "quy_ve_bank": [0] * n,
                "co_moc_tg": [0] * n,
            })

    chunks = pd.DataFrame({"content_text": ["x"], "doc_id": ["d"], "chunk_index": [0]})
    out = classify_chunks(chunks, _StubTopic(), StubCommitNo(), StubSpecCommit(),
                          spec_on_commitment=False)
    assert out["co_cam_ket"].iloc[0] == 1   # from LLM, not the PhoBERT pre-filter
    assert out["evidence"].iloc[0] == "ev"
    assert out["spec_level"].iloc[0] == 1
