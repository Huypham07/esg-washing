import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from esgwash.config import load_config
from esgwash.data.esgbert_labels import build_labeled_table


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tau",
        type=float,
        default=None,
        help="nguong confidence (mac dinh: cross_label.confidence trong topic.yml)",
    )
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument(
        "--force",
        action="store_true",
        help="tinh lai probs, bo cache",
    )

    args = ap.parse_args(argv)

    stats = build_labeled_table(
        load_config("topic"),
        batch_size=args.batch_size,
        tau=args.tau,
        force=args.force,
    )

    print(json.dumps(stats, indent=2, default=str))
    return stats


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
