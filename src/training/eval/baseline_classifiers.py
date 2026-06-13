"""Baseline classifiers cho bake-off (Phase 02b): majority + TF-IDF + LogisticRegression.

Chạy CPU, nhanh. Đọc data/vi_gold/<task>/{train,test}.parquet (schema: sentence, label 0/1).
Xuất outputs/baselines/<task>_baseline.json — CÙNG metric với train_model (macro_f1, f1_positive,
micro_f1) để compare_models gom vào bảng so sánh.

Chạy:  python -m src.training.eval.baseline_classifiers --tasks env soc gov
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

GOLD_DIR = Path("data/vi_gold")
OUT_DIR = Path("outputs/baselines")
SEED = 42


def _scores(y_true, y_pred) -> dict:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro")),
        "f1_positive": float(f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)),
    }


def _load(task: str, gold_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    tdir = gold_dir / task
    train_p, test_p = tdir / "train.parquet", tdir / "test.parquet"
    if not train_p.exists() or not test_p.exists():
        raise FileNotFoundError(f"Thiếu gold cho task '{task}': {train_p} / {test_p}")
    return pd.read_parquet(train_p), pd.read_parquet(test_p)


def run_task(task: str, gold_dir: Path = GOLD_DIR) -> dict:
    """Train 2 baseline trên 1 task -> dict metric (majority + tfidf_lr)."""
    train, test = _load(task, gold_dir)
    y_train, y_test = train["label"].astype(int), test["label"].astype(int)

    # Majority — đoán nhãn phổ biến nhất của train cho mọi câu test.
    majority_label = int(y_train.mode().iloc[0])
    y_major = [majority_label] * len(y_test)

    # TF-IDF (1-2 gram) + LogisticRegression (class_weight balanced cho lệch lớp).
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000)
    x_train = vec.fit_transform(train["sentence"].astype(str))
    x_test = vec.transform(test["sentence"].astype(str))
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    clf.fit(x_train, y_train)
    y_lr = clf.predict(x_test)

    return {
        "task": task,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "pos_rate_test": float(y_test.mean()),
        "majority": _scores(y_test, y_major),
        "tfidf_lr": _scores(y_test, y_lr),
    }


def main(args=None) -> None:
    parser = argparse.ArgumentParser(description="Baseline majority + TF-IDF+LR trên vi_gold")
    parser.add_argument("--tasks", nargs="+", default=["env", "soc", "gov"], help="Danh sách task")
    parser.add_argument("--gold-dir", type=str, default=str(GOLD_DIR))
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    args = parser.parse_args(args)

    gold_dir, out_dir = Path(args.gold_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for task in args.tasks:
        res = run_task(task, gold_dir)
        (out_dir / f"{task}_baseline.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        for name in ("majority", "tfidf_lr"):
            rows.append({"task": task, "model": name, **res[name]})
        print(f"[{task}] majority macro-F1 {res['majority']['macro_f1']:.3f} | "
              f"tfidf_lr macro-F1 {res['tfidf_lr']['macro_f1']:.3f} "
              f"(F1-pos {res['tfidf_lr']['f1_positive']:.3f}, pos-rate {res['pos_rate_test']:.2f})")

    print("\n=== Baseline summary ===")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nSaved -> {out_dir}/<task>_baseline.json")


if __name__ == "__main__":
    main()
