"""Baselines: TF-IDF+LR + XLM-R zero-shot -> experiments/metrics/baselines.json.

  python experiments/run_baselines.py [--skip-xlmr]
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from esgwash.config import load_config
from esgwash.data.topic_merge import PILLARS, carve_val
from esgwash.models.baselines import tfidf_lr_baseline, xlmr_zero_shot

COMMIT_HEADS = ["commitment"]


def _load(name: str, heads):
    train_full = pd.read_parquet(f"data/{name}_train.parquet")
    test = pd.read_parquet(f"data/{name}_test.parquet")
    mask = (train_full["source"] == "climatebert").to_numpy() if name == "commitment" else None
    train, val = carve_val(train_full, heads, mask=mask)
    return train, val, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-xlmr", action="store_true")
    args = ap.parse_args()
    results = {}

    topic_tr, topic_val, topic_te = _load("topic", list(PILLARS))
    com_tr, com_val, com_te = _load("commitment", COMMIT_HEADS)
    results["tfidf_lr_topic"] = tfidf_lr_baseline(topic_tr, topic_te, heads=list(PILLARS))
    results["tfidf_lr_commitment"] = tfidf_lr_baseline(com_tr, com_te, heads=COMMIT_HEADS)

    if not args.skip_xlmr:
        en = lambda d: d.assign(text=d["text_en"])  # noqa: E731
        cfg_t = {**load_config("topic"), "heads": list(PILLARS)}
        cfg_c = {**load_config("commitment_train"), "heads": COMMIT_HEADS}
        results["xlmr_zero_shot_topic"] = xlmr_zero_shot(
            cfg_t, en(topic_tr), en(topic_val), topic_te)
        results["xlmr_zero_shot_commitment"] = xlmr_zero_shot(
            cfg_c, en(com_tr), en(com_val), com_te)

    out = Path("experiments/metrics/baselines.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
