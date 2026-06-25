"""Test logic specificity scorer 3 muc (parse + retry + derive) khong tai model."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from esgwash.models.specificity_llm import SpecificityLLM, _extract_json, derive, derive_flags, enforce_evidence


def test_extract_json_strips_think_and_prose():
    assert _extract_json('rac<think>suy nghi</think> {"items": []} duoi') == {"items": []}
    assert _extract_json("khong co json") is None
    assert _extract_json('{"items": [}') is None       # JSON loi, khong salvage duoc item


def test_derive_3level():
    # Muc 2: dinh luong & quy ve chu the
    p, lvl = derive({"items": [{"is_quantified": True, "attributable_to_actor": True}]})
    assert (p, lvl) == (1.0, 2)
    # Muc 1: hanh dong co ten & quy ve chu the (khong so)
    p, lvl = derive({"items": [{"is_concrete_action": True, "attributable_to_actor": True}]})
    assert (p, lvl) == (0.5, 1)
    # Muc 0: so co nhung KHONG quy ve chu the (kieu BIDV/NHNN)
    p, lvl = derive({"items": [{"is_quantified": True, "attributable_to_actor": False}]})
    assert (p, lvl) == (0.0, 0)
    assert derive({"items": []}) == (0.0, 0)


class _StubLLM(SpecificityLLM):
    def __init__(self, replies):
        super().__init__({"retries": 2})
        self._replies = list(replies)

    def _complete(self, messages):
        return self._replies.pop(0)


def test_retry_then_success():
    # lan 1 rac -> lan 2 JSON hop le (concrete_action: khong bi verify_rubric huy)
    llm = _StubLLM(["rac khong json",
                    json.dumps({"items": [{"action_or_event": "trien khai B.One",
                                           "is_concrete_action": True,
                                           "attributable_to_actor": True}],
                                "has_baseline_or_timeline": False})])
    out = llm.score_one("BIDV trien khai B.One")
    assert out["parse_ok"] and out["is_specific"] == 1 and out["spec_level"] == 1


def test_fallback_after_retries():
    llm = _StubLLM(["x", "y", "z"])    # 1 + 2 retries deu hong
    out = llm.score_one("cau")
    assert out["parse_ok"] is False and out["is_specific"] == 0 and out["spec_level"] == 0


def test_verify_rubric_kills_fabricated_figure():
    # item is_quantified nhung figure '30%' KHONG co trong text -> huy -> Muc 0
    llm = _StubLLM([json.dumps({"items": [{"action_or_event": "giam phat thai",
                                           "figure": "30%", "is_quantified": True,
                                           "attributable_to_actor": True}],
                                "has_baseline_or_timeline": False})])
    out = llm.score_one("Ngan hang cam ket giam phat thai manh me")  # khong co so
    assert out["spec_level"] == 0 and out["is_specific"] == 0


def test_predict_columns():
    llm = _StubLLM([json.dumps({"items": [], "has_baseline_or_timeline": False})])
    df = llm.predict(["a"])
    assert list(df.columns) == ["p_specificity", "spec_level", "is_specific",
                                "parse_ok", "rubric", "raw"]


def test_classify_only_scores_commitments():
    """Specificity LLM chi cham tren chunk is_commitment=1 (tiet kiem + dung CTI)."""
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
            return pd.DataFrame({"p_specificity": [1.0] * n, "spec_level": [2] * n,
                                 "is_specific": [1] * n, "parse_ok": [True] * n,
                                 "rubric": ["{}"] * n, "raw": ["{}"] * n})

    chunks = pd.DataFrame({"content_text": ["cam ket A", "cau thuong B"],
                           "doc_id": ["d", "d"], "chunk_index": [0, 1]})
    spec = StubSpec()
    out = classify_chunks(chunks, StubTopic(), StubCommit(), spec)
    assert spec.calls == [["cam ket A"]]               # chi chunk commitment
    assert out.loc[0, "is_specific"] == 1 and out.loc[1, "is_specific"] == 0
    assert out.loc[0, "spec_level"] == 2


def test_derive_flags_levels():
    # Mức 2: có số định lượng & quy về chủ thể
    assert derive_flags({"co_so_dinh_luong": 1, "quy_ve_bank": 1}) == (1.0, 2)
    # Mức 2 hụt: có số nhưng KHÔNG quy về chủ thể -> rớt xuống theo hành động
    assert derive_flags({"co_so_dinh_luong": 1, "quy_ve_bank": 0,
                         "co_hanh_dong_ten": 1}) == (0.5, 1)
    # Mức 1: có hành động có tên, không số
    assert derive_flags({"co_hanh_dong_ten": 1}) == (0.5, 1)
    # Mức 0: không có gì
    assert derive_flags({}) == (0.0, 0)
    assert derive_flags({"co_so_dinh_luong": 1, "quy_ve_bank": 0}) == (0.0, 0)


def test_enforce_evidence_drops_unsupported_flag():
    text = "Ngân hàng triển khai hệ thống quản lý môi trường nội bộ."
    flags = {"co_cam_ket": 1, "co_hanh_dong_ten": 1, "co_so_dinh_luong": 1,
             "quy_ve_bank": 1, "co_moc_tg": 0}
    evidence = {"co_cam_ket": "triển khai", "co_hanh_dong_ten": "hệ thống quản lý môi trường",
                "co_so_dinh_luong": "5000 tỷ",  # KHÔNG có trong text -> phải hạ về 0
                "quy_ve_bank": "Ngân hàng"}
    out = enforce_evidence(flags, evidence, text)
    assert out["co_hanh_dong_ten"] == 1
    assert out["co_so_dinh_luong"] == 0   # evidence không phải substring
    assert out["co_cam_ket"] == 1
