"""Train M1 topic: Optuna -> best params -> train multi-seed with params.

  python scripts/train_topic.py --tune 30          # tune 30 trials (resumable) roi train
  python scripts/train_topic.py                    # train; tu dong doc best_params.json neu co
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from esgwash.config import load_config
from esgwash.data.topic_merge import PILLARS, carve_val
from esgwash.models.topic_model import TopicModel
from esgwash.models.trainer import multi_seed
from esgwash.models.tuning import apply_best_params, save_best_params, tune

TRAIN_PATH = Path("data/topic_train.parquet")
TEST_PATH = Path("data/topic_test.parquet")


def load_split(lang: str, val_seed: int = 42):
    train_full = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)
    if lang == "en":
        train_full = train_full.assign(text=train_full["text_en"])
        test = test.assign(text=test["text_en"])
    train, val = carve_val(train_full, PILLARS, seed=val_seed)
    return train, val, test


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="vi", choices=["vi", "en"])
    ap.add_argument("--tune", type=int, default=0, metavar="N",
                    help="so trial Optuna chay them truoc khi train (0 = bo qua)")
    args = ap.parse_args(argv)

    cfg = load_config("topic")
    cfg["heads"] = cfg["labels"]
    if args.lang == "en":
        cfg["backbone"] = cfg.get("en_backbone", "roberta-base")
        cfg["word_segment"] = False

    variant = args.lang
    out_dir = Path(cfg["output_dir"]) / variant
    best_path = out_dir / "best_params.json"

    train, val, test = load_split(args.lang)

    best = None
    if args.tune > 0:
        study = tune(cfg, train, val, n_trials=args.tune, study_dir=out_dir,
                     study_name=f"topic_{variant}", trainer_cls=TopicModel)
        best = save_best_params(study, best_path,
                                max_epochs=int(cfg.get("tune", {}).get("max_epochs", 10)))
        print(f"tune xong ({best['n_trials']} trials): val macro-F1 "
              f"{best['val_macro_f1']} | params -> {best_path}")
    elif best_path.exists():
        best = json.loads(best_path.read_text(encoding="utf-8"))
        print(f"dung best params da tune ({best['tuned_at']}, "
              f"val macro-F1 {best['val_macro_f1']})")
    else:
        print("CHUA TUNE — train voi default topic.yml "
              "(nen chay --tune 30 truoc khi lay ket qua chinh)")
    if best:
        cfg = apply_best_params(cfg, best)

    results = multi_seed(cfg, train, val, test, out_dir=out_dir, trainer_cls=TopicModel)
    results["tuned_params"] = best
    results["config"] = {"backbone": cfg["backbone"], "dropout": cfg.get("dropout", 0.1),
                         "train": cfg["train"], "variant": variant}

    metrics_path = Path(f"outputs/metrics/topic_{variant}.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["aggregate"], indent=2))


if __name__ == "__main__":
    main()
