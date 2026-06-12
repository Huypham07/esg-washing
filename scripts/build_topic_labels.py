"""Buoc chuan bi data: hoan thien nhan topic bang ESGBERT cross-inference.

  python scripts/build_topic_labels.py [--tau 0.9] [--batch-size 64] [--force]

Doc topic_masked.parquet, cache probs (esgbert_probs.parquet, --force de tinh lai),
dien o NaN-train confidence >= tau -> data/processed/gold/topic_labeled.parquet
(bang nhan dich, train_topic.py mac dinh dung). Tuong duong:
  python -m esgwash.pipeline.run --stage cross_label
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from esgwash.config import load_config
from esgwash.data.esgbert_labels import build_labeled_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=None,
                    help="nguong confidence (mac dinh: cross_label.confidence trong topic.yml)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--force", action="store_true", help="tinh lai probs, bo cache")
    args = ap.parse_args()

    stats = build_labeled_table(load_config("topic"), batch_size=args.batch_size,
                                tau=args.tau, force=args.force)
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
