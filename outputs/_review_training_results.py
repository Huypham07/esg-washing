"""Tổng hợp TOÀN BỘ kết quả training (gold + silver): confusion + per-class + macro/micro/f1_pos + baseline.

Chạy:  python outputs/_review_training_results.py
Nguồn: outputs/models/<dir>/test_predictions.parquet + phobert_test_summary.json (baseline+val).
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

GOLD = {"env": "topic_e", "soc": "topic_s", "gov": "topic_g",
        "commitment": "commitment", "specificity": "specificity"}
SILVER = {"env": "topic_e_silver", "soc": "topic_s_silver", "gov": "topic_g_silver",
          "commitment": "commitment_silver", "specificity": "specificity_silver"}


def analyze(model_dir: str):
    p = Path("outputs/models") / model_dir / "test_predictions.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    yt, yp = df.y_true.values, df.y_pred.values
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    pr, rc, f1c, _ = precision_recall_fscore_support(yt, yp, labels=[0, 1], zero_division=0)
    return {
        "n": len(df), "pos": int((yt == 1).sum()),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "p1": pr[1], "r1": rc[1], "f1pos": f1c[1],
        "p0": pr[0], "r0": rc[0], "f0": f1c[0],
        "macro": f1_score(yt, yp, average="macro"),
        "micro": f1_score(yt, yp, average="micro"),
    }


summ = json.loads(Path("outputs/compare/phobert_test_summary.json").read_text(encoding="utf-8"))


def block(title, mapping, with_base):
    print("=" * 92)
    print(title)
    print("=" * 92)
    hdr = f"{'task':12} {'n':>4} {'pos%':>5} {'val':>6} {'macro':>6} {'micro':>6} {'f1_POS':>6}"
    if with_base:
        hdr += f" {'majF1':>6} {'tfidf':>6} {'Δtfidf':>7}"
    print(hdr)
    print("-" * 92)
    for task, d in mapping.items():
        r = analyze(d)
        if r is None:
            print(f"{task:12} MISSING")
            continue
        posr = r["pos"] / r["n"]
        val = summ.get(task, {}).get("val_f1", float("nan")) if with_base else float("nan")
        line = f"{task:12} {r['n']:>4} {posr:>5.0%} {val:>6.3f} {r['macro']:>6.3f} {r['micro']:>6.3f} {r['f1pos']:>6.3f}"
        if with_base:
            base = summ.get(task, {}).get("baseline_test", {})
            maj, tf = base.get("majority", float("nan")), base.get("tfidf_lr", float("nan"))
            line += f" {maj:>6.3f} {tf:>6.3f} {r['macro']-tf:>+7.3f}"
        print(line)
    print("\n  -- confusion + per-class (lớp 1 = positive) --")
    for task, d in mapping.items():
        r = analyze(d)
        if r is None:
            continue
        print(f"  {task:12} | TP={r['tp']:>4} FP={r['fp']:>4} FN={r['fn']:>4} TN={r['tn']:>4}"
              f" | lớp1 P={r['p1']:.2f} R={r['r1']:.2f} F1={r['f1pos']:.2f}"
              f" | lớp0 P={r['p0']:.2f} R={r['r0']:.2f} F1={r['f0']:.2f}")
    print()


block("GOLD (vi_gold — expert dịch) — bảng chính", GOLD, with_base=True)
block("SILVER (vi_silver — nhãn LLM in-domain, test CIRCULAR)", SILVER, with_base=False)
