# experiments/rescore_eval_gold.py
"""Regenerate eval_gold_report.json bang cach RE-SCORE (P07 pre-flight fix).

Boi canh: eval_gold_report.json commit cu duoc sinh 26/06 tren gold relabel v1;
gold relabel bi cap nhat 28/06 (commit 1ff50ca) nhung report KHONG duoc sinh lai
-> bo so committed cu (QWK 0.27, ceiling self-join 1.0) KHONG tai lap duoc.

Script nay KHONG chay lai model (guardrail: giu nguyen du doan run B):
- doc du doan model da co:  experiments/eval/gold_classified.parquet  (bat bien)
- doc gold relabel HIEN TAI: data/gold_annot_{1,2}_relabeled.xlsx
- cham lai bang dung evaluate() cua eval_gold.py (mot nguon su that duy nhat)
- GATE: human ceiling QWK (A vs B) phai ~0.706, model-vs-A QWK ~0.66
  (khop ablation_metrics.json da duoc Kaggle c1_live validate 0/390 mismatch).

  python experiments/rescore_eval_gold.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from eval_gold import OUT, evaluate, load_gold, plot_within_band  # noqa: E402


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    clf_path = OUT / "gold_classified.parquet"
    clf = pd.read_parquet(clf_path)
    gold = load_gold()
    print(f"### rescore: {len(gold)} gold chunk x {len(clf)} predictions (run B, KHONG chay lai model)")

    report = evaluate(clf, gold, with_spec="spec_level" in clf.columns)
    report["_provenance"] = {
        "regenerated": "2026-07-11 P07 pre-flight (rescore only)",
        "predictions": "experiments/eval/gold_classified.parquet (run B, unchanged)",
        "gold": "data/gold_annot_{1,2}_relabeled.xlsx (relabel hien tai, commit 1ff50ca)",
        "scorer": "experiments/eval_gold.py::evaluate (shared single source of truth)",
    }

    # GATE: doi chieu voi ablation_metrics.json (da duoc Kaggle validate)
    spec = report["labels"].get("spec_level", {})
    ceiling = spec.get("human_ceiling_A_vs_B", {}).get("quadratic_weighted_kappa")
    qwk_a = spec.get("model_vs_A", {}).get("quadratic_weighted_kappa")
    qwk_b = spec.get("model_vs_B", {}).get("quadratic_weighted_kappa")
    print(f"GATE spec_level: model-vs-A QWK={qwk_a}  model-vs-B QWK={qwk_b}  ceiling A-vs-B={ceiling}")
    assert ceiling is not None and abs(ceiling - 0.706) < 0.01, f"ceiling {ceiling} lech 0.706 — DUNG, kiem tra gold"
    assert qwk_a is not None and abs(qwk_a - 0.663) < 0.01, f"QWK_A {qwk_a} lech 0.663 — DUNG, kiem tra parquet"

    (OUT / "eval_gold_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_within_band(report)
    print(f"-> {OUT / 'eval_gold_report.json'} (+ eval_gold_within_band.png) — GATE PASS")


if __name__ == "__main__":
    main()
