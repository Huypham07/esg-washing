"""Phase 01 (P3-L1) — đo P/R của VERIFIER (NLI) trên ML-Promise evidence-assessment.

Mục đích: sanity-check θ=0.7 (θ CHUNG) cho task corroboration + báo P/R/F1 ở EN & VI; KHÔNG set θ riêng.
SCOPE (xem phase-03): calibrate RIÊNG verifier (phán evidence-presence mức paragraph), KHÔNG phải
retrieval end-to-end — ML-Promise `text` đã chứa evidence inline. Dùng EN 400 + bản dịch VI (line-aligned),
DROP dòng evidence_status NaN, KHÔNG dùng cột timeline (dirty).
CAVEAT (unresolved Q1): hypothesis ở đây là GENERIC ("đoạn có bằng chứng"), khác hypothesis action-specific
của path L1 (premise=câu, hypothesis=action) — đây là sanity-check verifier, không transfer θ tuyệt đối.

CẦN GPU (nạp NLI model). Output: experiments/metrics/verifier_calibration.json.

  python experiments/calibrate_verifier.py [--nli-model <id>] [--lambda 0.5]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from esgwash.config import load_config          # noqa: E402
from esgwash.grounding.nli import NLIScorer      # noqa: E402

MODEL_2MIL7 = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
EN_CSV = Path("source_data/source_dataset/ml_promise/ml_promise.csv")
VI_CSV = Path("source_data/translate/ml_promise_vi.csv")
OUT = Path("experiments/metrics/verifier_calibration.json")
THETAS = [0.5, 0.7, 0.9]

# AIS-framed hypothesis (Yin 2019 NLI-as-classifier / Rashkin 2023 AIS): premise = paragraph.
HYP = {
    "en": "This paragraph provides concrete evidence that the commitment was actually carried out.",
    "vi": "Đoạn văn này cung cấp bằng chứng cụ thể cho thấy cam kết đã thực sự được thực hiện.",
}


def load_pairs() -> tuple[list[str], list[str], np.ndarray]:
    """EN 400 + VI line-aligned 1:1; drop evidence_status NaN/blank. -> (text_en, text_vi, y)."""
    en = pd.read_csv(EN_CSV)
    vi = pd.read_csv(VI_CSV)
    assert len(en) == len(vi), f"VI ({len(vi)}) lệch dòng EN ({len(en)})"
    mask = (en["lang"] == "en") & en["evidence_status"].isin(["Yes", "No"])
    y = (en.loc[mask, "evidence_status"] == "Yes").astype(int).to_numpy()
    return (en.loc[mask, "text"].astype(str).tolist(),
            vi.loc[mask, "text"].astype(str).tolist(), y)


def score(scorer: NLIScorer, texts: list[str], hyp: str, lam: float) -> np.ndarray:
    """score = P_entail - lam * P_contra (khớp support_score_l1 sẽ dùng ở path L1)."""
    probs = scorer.score_pairs([(t, hyp) for t in texts])
    return scorer.entail(probs) - lam * scorer.contra(probs)


def metrics_at(y: np.ndarray, s: np.ndarray) -> dict:
    out = {}
    for th in THETAS:
        p, r, f, _ = precision_recall_fscore_support(
            y, (s >= th).astype(int), average="binary", zero_division=0)
        out[f"theta={th}"] = {"P": round(float(p), 4), "R": round(float(r), 4),
                              "F1": round(float(f), 4)}
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    f1s = [precision_recall_fscore_support(y, (s >= t).astype(int),
           average="binary", zero_division=0)[2] for t in grid]
    out["theta_star_ref"] = {"theta": float(grid[int(np.argmax(f1s))]),
                             "F1": round(float(max(f1s)), 4)}
    return out


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Đo P/R verifier NLI trên ML-Promise (sanity-check θ=0.7)")
    ap.add_argument("--nli-model", default=MODEL_2MIL7)
    ap.add_argument("--lambda", dest="lam", type=float, default=0.5)
    args = ap.parse_args(argv)

    text_en, text_vi, y = load_pairs()
    print(f"ML-Promise EN: n={len(y)} (pos evidence={int(y.sum())}); model={args.nli_model} λ={args.lam}")

    cfg = {**load_config("grounding"), "nli_model": args.nli_model}
    scorer = NLIScorer(cfg)

    res = {
        "model": args.nli_model, "lambda": args.lam,
        "n": int(len(y)), "n_pos": int(y.sum()), "theta_shared": 0.7,
        "scope_caveat": "Verifier-only (paragraph evidence-presence). KHÔNG phải retrieval end-to-end; "
                        "hypothesis generic (khác action-specific của L1).",
        "en": metrics_at(y, score(scorer, text_en, HYP["en"], args.lam)),
        "vi": metrics_at(y, score(scorer, text_vi, HYP["vi"], args.lam)),
    }
    out = OUT.with_name(f"verifier_calib_{args.nli_model.split('/')[-1]}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"en@0.7": res["en"]["theta=0.7"], "vi@0.7": res["vi"]["theta=0.7"],
                      "theta*_en": res["en"]["theta_star_ref"],
                      "theta*_vi": res["vi"]["theta_star_ref"]}, indent=2, ensure_ascii=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
