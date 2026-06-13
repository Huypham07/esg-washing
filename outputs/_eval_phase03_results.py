"""Tổng hợp kết quả train Phase 03 (cách C, ngưỡng 0.5): số CHÍNH THỨC (mean±std) + sàn TF-IDF.
KHÔNG phụ thuộc kết quả cũ (an toàn sau khi xóa outputs/_archive_pre_phase03/).
Chạy: python outputs/_eval_phase03_results.py
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

TASKS = ["env", "soc", "gov", "commitment", "specificity"]
OUTDIR = {"env": "topic_e", "soc": "topic_s", "gov": "topic_g",
          "commitment": "commitment", "specificity": "specificity"}


def g(d, k, s):
    v = d.get(k, {})
    return v.get(s) if isinstance(v, dict) else v


print(f"{'task':13}{'macro-F1 (mean±std)':>22}{'f1_pos':>9}{'tfidf':>8}{'major':>8}{'seeds':>6}{'thr':>6}")
print("-" * 78)
for t in TASKS:
    ms = json.load(open(f"outputs/models/{OUTDIR[t]}/metrics_summary.json", encoding="utf-8"))
    te = ms["test"]
    nm, nms = g(te, "macro_f1", "mean"), g(te, "macro_f1", "std")
    nfp = g(te, "f1_positive", "mean")
    tr = pd.read_parquet(f"data/vi_gold/{t}/train.parquet")
    ted = pd.read_parquet(f"data/vi_gold/{t}/test.parquet")
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    Xtr, Xte = vec.fit_transform(tr.sentence), vec.transform(ted.sentence)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, tr.label)
    maj = DummyClassifier(strategy="most_frequent").fit(Xtr, tr.label)
    tf = f1_score(ted.label, lr.predict(Xte), average="macro")
    mj = f1_score(ted.label, maj.predict(Xte), average="macro")
    print(f"{t:13}{nm:.3f} ± {nms:.3f}{'':6}{nfp:>8.3f}{tf:>8.3f}{mj:>8.3f}"
          f"{ms.get('n_seeds'):>6}{ms.get('inference_threshold'):>6}")

print("\n=== classification report (seed0) — task yếu ===")
for t in ["gov", "commitment", "specificity"]:
    ms = json.load(open(f"outputs/models/{OUTDIR[t]}/metrics_summary.json", encoding="utf-8"))
    p0 = ms["per_seed"][0]
    print(f"\n--- {t} (seed {p0['seed']}) ---")
    print(p0.get("report", "(no report)"))
