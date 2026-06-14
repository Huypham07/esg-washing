"""Test logic specificity scorer (parse JSON + retry + derive luat) khong tai model."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from esgwash.models.specificity_llm import SpecificityLLM, _extract_json, derive


def test_extract_json_strips_think_and_prose():
    assert _extract_json("rac<think>suy nghi</think> oke {\"items\": []} duoi") == {"items": []}
    assert _extract_json("khong co json") is None
    assert _extract_json('{"items": [}') is None  # JSON loi -> None


def test_derive_rule():
    # co dai luong dinh luong quy ve chu the -> is_specific=1, diem cao
    r1 = {"items": [{"is_quantified": True, "attributable_to_actor": True}],
          "has_baseline_or_timeline": True}
    p, s = derive(r1)
    assert s == 1 and p == 1.0
    # so co nhung khong quy ve chu the (kieu BIDV) -> is_specific=0
    r2 = {"items": [{"is_quantified": True, "attributable_to_actor": False}],
          "has_baseline_or_timeline": False}
    p, s = derive(r2)
    assert s == 0 and p == 0.2          # chi any_quantified
    # khong so lieu gi -> 0
    p, s = derive({"items": [{"is_quantified": False, "attributable_to_actor": True}]})
    assert s == 0 and p == 0.0


class _StubLLM(SpecificityLLM):
    def __init__(self, replies):
        super().__init__({"retries": 2})
        self._replies = list(replies)

    def _complete(self, messages):
        return self._replies.pop(0)


def test_retry_then_success():
    llm = _StubLLM(["rac khong json",
                    json.dumps({"items": [{"is_quantified": True,
                                           "attributable_to_actor": True}],
                                "has_baseline_or_timeline": False})])
    out = llm.score_one("cau gi do")
    assert out["parse_ok"] and out["is_specific"] == 1


def test_fallback_after_retries():
    llm = _StubLLM(["x", "y", "z"])   # 1 + 2 retries deu hong
    out = llm.score_one("cau")
    assert out["parse_ok"] is False and out["is_specific"] == 0 and out["p_specificity"] == 0.0


def test_predict_columns():
    llm = _StubLLM([json.dumps({"items": [], "has_baseline_or_timeline": False})])
    df = llm.predict(["a"])
    assert list(df.columns) == ["p_specificity", "is_specific", "parse_ok", "rubric"]


def test_classify_only_scores_commitments():
    """Specificity LLM chi cham tren cau is_commitment=1 (tiet kiem + dung CTI)."""
    from esgwash.pipeline.inference import classify_sentences

    class StubTopic:
        def predict(self, texts):
            n = len(texts)
            return pd.DataFrame({"env": [0.9] * n, "soc": [0.1] * n, "gov": [0.1] * n,
                                 "is_env": [1] * n, "is_soc": [0] * n, "is_gov": [0] * n,
                                 "pillar": ["env"] * n})

    class StubCommit:
        def predict(self, texts):
            n = len(texts)
            return pd.DataFrame({"p_commitment": [0.9, 0.1][:n] + [0.1] * (n - 2),
                                 "is_commitment": [1, 0][:n] + [0] * (n - 2)})

    class StubSpec:
        def __init__(self):
            self.calls = []

        def predict(self, texts):
            self.calls.append(list(texts))
            n = len(texts)
            return pd.DataFrame({"p_specificity": [1.0] * n, "is_specific": [1] * n,
                                 "parse_ok": [True] * n, "rubric": ["{}"] * n})

    sents = pd.DataFrame({"sentence": ["cam ket A", "cau thuong B"],
                          "doc_id": ["d", "d"], "sent_id": [0, 1]})
    spec = StubSpec()
    out = classify_sentences(sents, StubTopic(), StubCommit(), spec)
    assert spec.calls == [["cam ket A"]]          # chi cau commitment
    assert out.loc[0, "is_specific"] == 1 and out.loc[1, "is_specific"] == 0
