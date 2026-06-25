"""Test logic specificity scorer 4 co atomic specificity (parse + retry + derive) khong tai model."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from esgwash.models.specificity_llm import (
    ATOMIC_FLAGS,
    SpecificityLLM,
    _extract_json_obj,
    _parse_flags,
    derive_flags,
    enforce_evidence,
    verify_rubric_flags,
)


def test_extract_json_obj_strips_think_and_prose():
    # strips <think> and returns first balanced JSON object
    obj = _extract_json_obj('rac<think>suy nghi</think> {"co_cam_ket": true} duoi')
    assert obj == {"co_cam_ket": True}
    assert _extract_json_obj("khong co json") is None
    assert _extract_json_obj('{"co_cam_ket": [}') is None   # invalid JSON -> None


class _StubLLM(SpecificityLLM):
    def __init__(self, replies):
        super().__init__({"retries": 2})
        self._replies = list(replies)

    def _complete(self, messages):
        return self._replies.pop(0)


# ── Task 3 new tests ──────────────────────────────────────────────────────────

def test_score_one_emits_atomic_flags():
    text = "Ngân hàng sẽ triển khai hệ thống quản lý môi trường nội bộ."
    reply = json.dumps({"co_hanh_dong_ten": True,
                        "co_so_dinh_luong": False, "quy_ve_bank": True, "co_moc_tg": False,
                        "evidence": {"co_hanh_dong_ten": "hệ thống quản lý môi trường nội bộ",
                                     "quy_ve_bank": "Ngân hàng"}}, ensure_ascii=False)
    out = _StubLLM([reply]).score_one(text)
    assert out["parse_ok"] is True
    assert "co_cam_ket" not in out
    assert out["co_hanh_dong_ten"] == 1
    assert out["co_so_dinh_luong"] == 0
    assert out["spec_level"] == 1 and out["is_specific"] == 1


def test_score_one_fabricated_quant_dropped():
    text = "Ngân hàng cam kết giảm phát thải mạnh mẽ."   # khong co so
    reply = json.dumps({"co_so_dinh_luong": True, "quy_ve_bank": True,
                        "co_hanh_dong_ten": False, "co_moc_tg": False,
                        "evidence": {"co_so_dinh_luong": "giảm 30%",  # so khong co trong text
                                     "quy_ve_bank": "Ngân hàng"}}, ensure_ascii=False)
    out = _StubLLM([reply]).score_one(text)
    assert "co_cam_ket" not in out
    assert out["co_so_dinh_luong"] == 0 and out["spec_level"] == 0


def test_score_one_parse_fail_safe():
    out = _StubLLM(["rac", "van rac", "rac nua"]).score_one("cau")
    assert out["parse_ok"] is False and out["spec_level"] == 0
    assert "co_cam_ket" not in out
    assert out["co_hanh_dong_ten"] == 0


def test_score_one_salvages_truncated_json():
    # Qwen3-0.6B truncates mid-JSON: booleans + 2 complete evidence pairs survive,
    # rest cut off (unbalanced braces -> _extract_json_obj fails -> salvage kicks in).
    text = "Ngân hàng sẽ triển khai hệ thống quản lý môi trường nội bộ."
    truncated = (
        '{"co_hanh_dong_ten": true, "co_so_dinh_luong": false, '
        '"quy_ve_bank": true, "co_moc_tg": false, '
        '"evidence": {"co_hanh_dong_ten": "hệ thống quản lý môi trường nội bộ", '
        '"quy_ve_bank": "Ngân '  # truncated mid-string, unbalanced
    )
    out = _StubLLM([truncated]).score_one(text)
    assert out["parse_ok"] is True
    assert "co_cam_ket" not in out
    assert out["co_hanh_dong_ten"] == 1
    assert out["spec_level"] == 1


# ── Retry behaviour ───────────────────────────────────────────────────────────

def test_retry_then_success():
    # lan 1 rac -> lan 2 JSON hop le (hanh dong co ten -> Muc 1)
    text = "BIDV trien khai B.One"
    reply = json.dumps({"co_hanh_dong_ten": True,
                        "co_so_dinh_luong": False, "quy_ve_bank": True, "co_moc_tg": False,
                        "evidence": {"co_hanh_dong_ten": "B.One",
                                     "quy_ve_bank": "BIDV"}}, ensure_ascii=False)
    llm = _StubLLM(["rac khong json", reply])
    out = llm.score_one(text)
    assert "co_cam_ket" not in out
    assert out["parse_ok"] and out["is_specific"] == 1 and out["spec_level"] == 1


def test_fallback_after_retries():
    llm = _StubLLM(["x", "y", "z"])    # 1 + 2 retries deu hong
    out = llm.score_one("cau")
    assert out["parse_ok"] is False and out["is_specific"] == 0 and out["spec_level"] == 0


# ── verify_rubric_flags ───────────────────────────────────────────────────────

def test_verify_rubric_flags_kills_fabricated_figure():
    # co_so_dinh_luong=1 nhung so trong evidence khong co trong text -> ha ve 0
    flags = {f: 0 for f in ATOMIC_FLAGS}
    flags["co_so_dinh_luong"] = 1
    flags["quy_ve_bank"] = 1
    evidence = {"co_so_dinh_luong": "30%"}
    text = "Ngan hang cam ket giam phat thai manh me"  # khong co so
    out = verify_rubric_flags(flags, evidence, text)
    assert out["co_so_dinh_luong"] == 0


def test_verify_rubric_flags_keeps_real_figure():
    flags = {f: 0 for f in ATOMIC_FLAGS}
    flags["co_so_dinh_luong"] = 1
    evidence = {"co_so_dinh_luong": "30%"}
    text = "giam 30% phat thai"  # so co that
    out = verify_rubric_flags(flags, evidence, text)
    assert out["co_so_dinh_luong"] == 1


# ── predict columns ───────────────────────────────────────────────────────────

def test_predict_columns():
    text = "Ngân hàng triển khai hệ thống B.One."
    reply = json.dumps({"co_hanh_dong_ten": True,
                        "co_so_dinh_luong": False, "quy_ve_bank": True, "co_moc_tg": False,
                        "evidence": {"co_hanh_dong_ten": "hệ thống B.One",
                                     "quy_ve_bank": "Ngân hàng"}}, ensure_ascii=False)
    llm = _StubLLM([reply])
    df = llm.predict([text])
    expected = ["p_specificity", "spec_level", "is_specific", "parse_ok",
                "rubric", "raw", "evidence", *ATOMIC_FLAGS]
    assert list(df.columns) == expected
    # co_cam_ket must not appear (only 4 specificity flags)
    assert "co_cam_ket" not in df.columns


# ── Tasks 1+2 tests (keep passing) ───────────────────────────────────────────

def test_derive_flags_levels():
    # Muc 2: co so dinh luong & quy ve chu the
    assert derive_flags({"co_so_dinh_luong": 1, "quy_ve_bank": 1}) == (1.0, 2)
    # Muc 2 hut: co so nhung KHONG quy ve chu the -> rot xuong theo hanh dong
    assert derive_flags({"co_so_dinh_luong": 1, "quy_ve_bank": 0,
                         "co_hanh_dong_ten": 1}) == (0.5, 1)
    # Muc 1: co hanh dong co ten, khong so
    assert derive_flags({"co_hanh_dong_ten": 1}) == (0.5, 1)
    # Muc 0: khong co gi
    assert derive_flags({}) == (0.0, 0)
    assert derive_flags({"co_so_dinh_luong": 1, "quy_ve_bank": 0}) == (0.0, 0)


def test_enforce_evidence_drops_unsupported_flag():
    text = "Ngân hàng triển khai hệ thống quản lý môi trường nội bộ."
    flags = {"co_hanh_dong_ten": 1, "co_so_dinh_luong": 1,
             "quy_ve_bank": 1, "co_moc_tg": 0}
    evidence = {"co_hanh_dong_ten": "hệ thống quản lý môi trường",
                "co_so_dinh_luong": "5000 tỷ",  # KHONG co trong text -> phai ha ve 0
                "quy_ve_bank": "Ngân hàng"}
    out = enforce_evidence(flags, evidence, text)
    assert out["co_hanh_dong_ten"] == 1
    assert out["co_so_dinh_luong"] == 0   # evidence khong phai substring
    assert "co_cam_ket" not in out        # co_cam_ket removed from pipeline
    assert out["quy_ve_bank"] == 1   # evidence "Ngân hàng" IS a substring
    assert out["co_moc_tg"] == 0     # flag input was 0 -> stays 0


# ── run.py integration smoke-test ─────────────────────────────────────────────

def test_classify_only_scores_commitments():
    """Specificity LLM chi cham tren chunk is_commitment=1 (tiet kiem + dung CTI).
    Final is_commitment comes from PhoBERT (StubCommit) gated by ESG, NOT from co_cam_ket."""
    from esgwash.run import classify_chunks

    class StubTopic:
        def predict(self, texts):
            n = len(texts)
            return pd.DataFrame({"env": [0.9] * n, "soc": [0.1] * n, "gov": [0.1] * n,
                                 "is_env": [1] * n, "is_soc": [0] * n, "is_gov": [0] * n,
                                 "pillar": ["env"] * n})

    class StubCommit:
        def predict(self, texts):
            return pd.DataFrame({"p_commitment": [0.9, 0.1], "is_commitment": [1, 0]})

    class StubSpec:
        def __init__(self):
            self.calls = []

        def predict(self, texts):
            self.calls.append(list(texts))
            n = len(texts)
            return pd.DataFrame({
                "p_specificity": [1.0] * n, "spec_level": [2] * n,
                "is_specific": [1] * n, "parse_ok": [True] * n,
                "rubric": ["{}"] * n, "raw": ["{}"] * n,
                "evidence": ["{}"] * n,
                **{f: [0] * n for f in ATOMIC_FLAGS},
            })

    chunks = pd.DataFrame({"content_text": ["cam ket A", "cau thuong B"],
                           "doc_id": ["d", "d"], "chunk_index": [0, 1]})
    spec = StubSpec()
    out = classify_chunks(chunks, StubTopic(), StubCommit(), spec)
    assert spec.calls == [["cam ket A"]]               # chi chunk commitment
    assert out.loc[0, "is_specific"] == 1 and out.loc[1, "is_specific"] == 0
    assert out.loc[0, "spec_level"] == 2
    # Final is_commitment gate: PhoBERT(row0)=1 AND is_env=1 -> 1
    assert out.loc[0, "is_commitment"] == 1
    # PhoBERT(row1)=0 -> final 0 regardless of ESG
    assert out.loc[1, "is_commitment"] == 0
    # co_cam_ket must NOT appear in output columns
    assert "co_cam_ket" not in out.columns
