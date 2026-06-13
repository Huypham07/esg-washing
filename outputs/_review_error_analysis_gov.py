"""Error analysis gov (gold) — đọc câu G bị BỎ SÓT (FN) và BÁO NHẦM (FP).

Chạy:  python outputs/_review_error_analysis_gov.py
Nguồn: outputs/models/topic_g/test_predictions.parquet (cols: sentence, y_true, y_pred).
"""
import os
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)


def bucketize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cond = [
        (df.y_true == 1) & (df.y_pred == 1),
        (df.y_true == 0) & (df.y_pred == 0),
        (df.y_true == 1) & (df.y_pred == 0),
        (df.y_true == 0) & (df.y_pred == 1),
    ]
    df["bucket"] = "?"
    for c, name in zip(cond, ["TP", "TN", "FN", "FP"]):
        df.loc[c, "bucket"] = name
    df["nword"] = df.sentence.str.split().str.len()
    return df


def top_words(series: pd.Series, k: int = 15) -> list:
    stop = set("và của là các có được trong cho với để theo về một những đã sẽ này đó "
               "the of and to in a for on is are we our by that as at".split())
    c = Counter()
    for s in series:
        for w in re.findall(r"[A-Za-zÀ-ỹ]+", str(s).lower()):
            if len(w) > 2 and w not in stop:
                c[w] += 1
    return c.most_common(k)


df = bucketize(pd.read_parquet("outputs/models/topic_g/test_predictions.parquet"))
vc = df.bucket.value_counts()
TP, TN, FN, FP = (int(vc.get(x, 0)) for x in ["TP", "TN", "FN", "FP"])
prec = TP / (TP + FP) if TP + FP else 0
rec = TP / (TP + FN) if TP + FN else 0

print("=" * 78)
print(f"GOV (gold)  test n={len(df)}  | TP={TP} FN={FN} FP={FP} TN={TN}")
print(f"precision={prec:.3f}  recall={rec:.3f}  (FN = {FN}/{TP+FN} positive bị bỏ sót)")
print("=" * 78)

print("\n-- Độ dài câu (số từ) theo nhóm --")
for b in ["TP", "FN", "FP", "TN"]:
    sub = df[df.bucket == b]
    if len(sub):
        print(f"  {b}: n={len(sub):3d} | median={sub.nword.median():.0f}  mean={sub.nword.mean():.1f}  "
              f"min={sub.nword.min()} max={sub.nword.max()}")

print("\n-- Top từ trong FN (câu G bị bỏ sót) vs TP (câu G bắt được) --")
print(f"  FN: {top_words(df[df.bucket=='FN'].sentence)}")
print(f"  TP: {top_words(df[df.bucket=='TP'].sentence)}")

print("\n" + "#" * 78)
print(f"# FN — {FN} câu G THẬT nhưng model đoán KHÔNG-G (bỏ sót)")
print("#" * 78)
for n, (_, r) in enumerate(df[df.bucket == "FN"].sort_values("nword").iterrows(), 1):
    print(f"\n[{n:2d}] ({r.nword} từ)\n     {r.sentence}")

print("\n" + "#" * 78)
print(f"# FP — {FP} câu KHÔNG-G nhưng model đoán G (báo nhầm)")
print("#" * 78)
for n, (_, r) in enumerate(df[df.bucket == "FP"].sort_values("nword").iterrows(), 1):
    print(f"\n[{n:2d}] ({r.nword} từ)\n     {r.sentence}")
