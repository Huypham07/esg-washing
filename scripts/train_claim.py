"""Train M2 claim (multi-task commitment + specificity) theo configs/claim.yml.

  python scripts/train_claim.py --tune 30          # tune Optuna (resumable) roi train
  python scripts/train_claim.py                    # train; tu dong doc best_params.json neu co
  python scripts/train_claim.py --single-task      # chay them fallback 2 model rieng (ablation)
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from esgwash.config import load_config
from esgwash.data import claim_merge
from esgwash.models.claim_model import MAIN_HEADS, ClaimModel, run_single_task_fallback
from esgwash.models.trainer import multi_seed
from esgwash.models.tuning import apply_best_params, save_best_params, tune

GOLD = Path("data/processed/gold/claim_table.parquet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="vi", choices=["vi", "en"])
    ap.add_argument("--tune", type=int, default=0, metavar="N",
                    help="so trial Optuna chay them truoc khi train (0 = bo qua)")
    ap.add_argument("--single-task", action="store_true")
    args = ap.parse_args()

    cfg = load_config("claim")
    cfg["heads"] = list(MAIN_HEADS)
    if args.lang == "vi" and GOLD.exists():
        df = pd.read_parquet(GOLD)
    else:
        df = claim_merge.build_claim_table(
            lang=args.lang,
            augment_action=cfg.get("augment_action_500", True),
            augment_ml_promise=cfg.get("augment_ml_promise", True))
    if args.lang == "en":
        cfg["backbone"] = cfg.get("en_backbone", "roberta-base")
        cfg["word_segment"] = False

    train = df[df["split"] == "train"]
    dev = df[(df["split"] == "dev") & (df["source"] == "climatebert")]
    test = df[df["split"] == "test"]
    out_dir = Path(cfg["output_dir"]) / args.lang
    best_path = out_dir / "best_params.json"

    best = None
    if args.tune > 0:
        study = tune(cfg, train, dev, n_trials=args.tune, study_dir=out_dir,
                     study_name=f"claim_{args.lang}", trainer_cls=ClaimModel)
        best = save_best_params(study, best_path,
                                max_epochs=int(cfg.get("tune", {}).get("max_epochs", 10)))
        print(f"tune xong ({best['n_trials']} trials): dev macro-F1 "
              f"{best['dev_macro_f1']} | params -> {best_path}")
    elif best_path.exists():
        best = json.loads(best_path.read_text(encoding="utf-8"))
        print(f"dung best params da tune ({best['tuned_at']}, "
              f"dev macro-F1 {best['dev_macro_f1']})")
    else:
        print("CHUA TUNE — train voi default claim.yml "
              "(nen chay --tune 30 truoc khi lay ket qua chinh)")
    if best:
        cfg = apply_best_params(cfg, best)

    results = {"tuned_params": best,
               "multi_task": multi_seed(cfg, train, dev, test, out_dir=out_dir,
                                        trainer_cls=ClaimModel)}

    if args.single_task:
        base = df[df["source"] == "climatebert"]
        results["single_task"] = run_single_task_fallback(
            cfg, base[base["split"] == "train"], dev, base[base["split"] == "test"])

    metrics_path = Path(f"outputs/metrics/claim_{args.lang}.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["multi_task"]["aggregate"], indent=2))


if __name__ == "__main__":
    main()
