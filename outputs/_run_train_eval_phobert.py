"""Phase 02 — train+eval 5 PhoBERT classifiers (auto-load best_params) + baseline. Tương đương nb03.
Chạy: .venv/Scripts/python.exe outputs/_run_train_eval_phobert.py
Xuất: outputs/models/<dir>/{final, metrics_summary.json, test_predictions.parquet} + outputs/compare/phobert_test_summary.json
"""
import sys, os; sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import warnings; warnings.filterwarnings("ignore")
import json, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]      # repo root (cha của outputs/)
os.chdir(ROOT); sys.path.insert(0, str(ROOT))   # cho `from src...` + path tương đối chạy mọi nơi
from src.training.train_model import load_yaml_config, resolve_runtime_config, run

RAW = load_yaml_config(Path("config/train.yml"))
TASKS = ["env", "soc", "gov", "commitment", "specificity"]
OUTDIR = {"env": "topic_e", "soc": "topic_s", "gov": "topic_g",
          "commitment": "commitment", "specificity": "specificity"}

VAL_F1 = {"env": 0.9532, "soc": 0.9274, "gov": 0.8528, "commitment": 0.9171, "specificity": 0.9104}


def _m(test: dict, k: str) -> float:
    """mean của metric — schema multi-seed (Phase 02) là {mean,std}; fallback float cũ."""
    v = test.get(k)
    return float(v.get("mean", 0)) if isinstance(v, dict) else float(v or 0)


def _s(test: dict, k: str) -> float:
    v = test.get(k)
    return float(v.get("std", 0)) if isinstance(v, dict) else 0.0


results = {}
t0 = time.time()
for t in TASKS:
    print(f"\n{'#'*60}\n# TRAIN {t}\n{'#'*60}")
    cfg = resolve_runtime_config(RAW, task=t)
    bp_path = Path(f"outputs/models/{OUTDIR[t]}/best_params_{t}.json")
    if bp_path.exists():
        bp = json.load(open(bp_path))
        if "max_length" in bp:
            cfg["model"]["max_length"] = bp.pop("max_length")
        cfg["training"].update(bp)
        print(f"[{t}] best_params: {bp}")
    else:
        print(f"[{t}] NO best_params -> default HP")
    ts = time.time()
    m = run(cfg)
    results[t] = m.get("test", {})
    print(f"[{t}] done in {time.time()-ts:.0f}s -> test macro_f1={_m(results[t],'macro_f1'):.4f}±{_s(results[t],'macro_f1'):.3f}")

# --- baseline test (majority + TF-IDF+LR) cho cả 5 ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score

baselines = {}
for t in TASKS:
    tr = pd.read_parquet(f"data/vi_gold/{t}/train.parquet")
    te = pd.read_parquet(f"data/vi_gold/{t}/test.parquet")
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    Xtr, Xte = vec.fit_transform(tr["sentence"]), vec.transform(te["sentence"])
    lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, tr["label"])
    maj = DummyClassifier(strategy="most_frequent").fit(Xtr, tr["label"])
    baselines[t] = {
        "majority": float(f1_score(te["label"], maj.predict(Xte), average="macro")),
        "tfidf_lr": float(f1_score(te["label"], lr.predict(Xte), average="macro")),
    }

summary = {t: {"phobert_test": results[t], "baseline_test": baselines[t], "val_f1": VAL_F1[t]} for t in TASKS}
out = Path("outputs/compare/phobert_test_summary.json")
out.parent.mkdir(parents=True, exist_ok=True)
json.dump(summary, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

print("\n" + "=" * 78)
print(f"{'task':12} {'major':>7} {'tfidf':>7} {'PhoBERT±std':>15} {'f1_pos±std':>14} {'valF1':>7}")
for t in TASKS:
    ph = _m(results[t], "macro_f1"); phs = _s(results[t], "macro_f1")
    bl = baselines[t]["tfidf_lr"]; mj = baselines[t]["majority"]
    fp = _m(results[t], "f1_positive"); fps = _s(results[t], "f1_positive"); vf = VAL_F1[t]
    print(f"{t:12} {mj:7.3f} {bl:7.3f} {ph:7.3f}±{phs:.3f}  {fp:6.3f}±{fps:.3f}  {vf:7.4f}")
print("=" * 78)
print(f"Total {time.time()-t0:.0f}s. Saved -> {out}")
