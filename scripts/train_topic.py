"""Train M1 topic theo configs/topic.yml (multi-seed).

  python scripts/train_topic.py [--lang vi|en] [--quick]
--quick: 1 seed, de smoke-test truoc khi chay full.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from esgwash.config import load_config
from esgwash.data import topic_merge
from esgwash.models.topic_model import TopicModel
from esgwash.models.trainer import multi_seed

GOLD = Path("data/processed/gold/topic_masked.parquet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="vi", choices=["vi", "en"])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    cfg = load_config("topic")
    cfg["heads"] = cfg["labels"]
    if args.quick:
        cfg["train"]["seeds"] = cfg["train"]["seeds"][:1]

    if args.lang == "vi" and GOLD.exists():
        df = pd.read_parquet(GOLD)
    else:
        df = topic_merge.build_masked_table(lang=args.lang)
        df = topic_merge.split_stratified(df, seed=42)
        if cfg.get("augment_env_claims_positives", False):
            df = topic_merge.augment_env_claims(df, lang=args.lang)
    if args.lang == "en":
        cfg["backbone"] = cfg.get("en_backbone", "roberta-base")
        cfg["word_segment"] = False

    train = df[df["split"] == "train"]
    dev = df[df["split"] == "dev"]
    test = df[df["split"] == "test"]
    out_dir = Path(cfg["output_dir"]) / args.lang
    results = multi_seed(cfg, train, dev, test, out_dir=out_dir, trainer_cls=TopicModel)

    metrics_path = Path(f"outputs/metrics/topic_{args.lang}.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["aggregate"], indent=2))


if __name__ == "__main__":
    main()
