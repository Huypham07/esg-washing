"""Phase 01 diagnostic — verifier FACTUAL test trên ML-Promise JA (cô lập meta-hypothesis vs verifier).

Calibrate generic (meta-hypothesis "đoạn có bằng chứng") cho θ*≈0.05 — nghi do HYPOTHESIS meta, không
phải verifier kém. Test này dùng cặp FACTUAL THẬT — đúng cách verifier hoạt động ở L1:
  REAL    : (premise=evidence_string, hypothesis=promise_string)  cùng dòng -> kỳ vọng P_entail CAO
  CONTROL : (premise=evidence_string, hypothesis=promise_string dòng KHÁC)  -> kỳ vọng P_entail THẤP
REAL >> CONTROL và REAL median KHÔNG gần-0  =>  verifier ỔN, meta-hypothesis là thủ phạm
=> Phase 02 dùng hypothesis action-specific là đúng hướng (calibrate θ_L1 từ B2 thật).
(Dùng JA vì chỉ JA có promise_string + evidence_string; model 2mil7 hỗ trợ JA.)

CẦN GPU (2mil7 đã cache). Output: experiments/metrics/verifier_ja_diagnostic.json.
  python experiments/diagnose_verifier_ja.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from esgwash.config import load_config          # noqa: E402
from esgwash.grounding.nli import NLIScorer      # noqa: E402

MODEL_2MIL7 = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
JA = Path("source_data/source_dataset/ml_promise/Trainset_Japanese.json")
OUT = Path("experiments/metrics/verifier_ja_diagnostic.json")


def load_pairs() -> list[tuple[str, str]]:
    """(evidence_string, promise_string) cho dòng có cả hai non-empty."""
    rows = json.loads(JA.read_text(encoding="utf-8-sig"))
    return [(str(r["evidence_string"]).strip(), str(r["promise_string"]).strip())
            for r in rows
            if str(r.get("evidence_string") or "").strip()
            and str(r.get("promise_string") or "").strip()]


def stats(a: np.ndarray) -> dict:
    a = np.asarray(a, dtype=float)
    return {"mean": round(float(a.mean()), 4), "median": round(float(np.median(a)), 4),
            "p25": round(float(np.percentile(a, 25)), 4),
            "p75": round(float(np.percentile(a, 75)), 4),
            "frac>=0.7": round(float((a >= 0.7).mean()), 4),
            "frac>=0.5": round(float((a >= 0.5).mean()), 4)}


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--nli-model", default=MODEL_2MIL7)
    model = ap.parse_args(argv).nli_model
    pairs = load_pairs()
    ev = [e for e, _ in pairs]
    pr = [p for _, p in pairs]
    pr_ctrl = pr[1:] + pr[:1]  # roll 1 -> claim lệch dòng, deterministic (không cần random)
    print(f"JA pairs (evidence+promise non-empty): n={len(pairs)}; model={model}")

    scorer = NLIScorer({**load_config("grounding"), "nli_model": model})
    real = scorer.entail(scorer.score_pairs(list(zip(ev, pr))))
    ctrl = scorer.entail(scorer.score_pairs(list(zip(ev, pr_ctrl))))

    res = {
        "model": model, "n_pairs": len(pairs),
        "real_evidence_to_promise": stats(real),
        "control_mismatched_claim": stats(ctrl),
        "paired_real_gt_control": round(float((real > ctrl).mean()), 4),
        "generic_calibration_theta_star_ref": 0.05,
    }
    out = OUT.with_name(f"verifier_ja_diagnostic_{model.split('/')[-1]}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
