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
from esgwash.data.topic_merge import carve_val
from esgwash.models.claim_model import MAIN_HEADS, ClaimModel, run_single_task_fallback
from esgwash.models.trainer import multi_seed
from esgwash.models.tuning import apply_best_params, save_best_params, tune

TRAIN_PATH = Path("data/claim_train.parquet")
TEST_PATH = Path("data/claim_test.parquet")


def load_split(lang: str, val_seed: int = 42):
    """Doc 2 file phang (khong con cot split); cat val tu train luc train.

    val chi lay tu climatebert de cung phan phoi voi test (augment action/ml_promise
    chi o train). Khong leak: moi dong la 1 text duy nhat.
    """
    train_full = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)
    if lang == "en":
        train_full = train_full.assign(text=train_full["text_en"])
        test = test.assign(text=test["text_en"])
    cb = (train_full["source"] == "climatebert").to_numpy()
    train, val = carve_val(train_full, list(MAIN_HEADS), seed=val_seed, mask=cb)
    return train, val, test


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="vi", choices=["vi", "en"])
    ap.add_argument("--tune", type=int, default=0, metavar="N",
                    help="so trial Optuna chay them truoc khi train (0 = bo qua)")
    ap.add_argument("--single-task", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config("claim")
    cfg["heads"] = list(MAIN_HEADS)
    if args.lang == "en":
        cfg["backbone"] = cfg.get("en_backbone", "roberta-base")
        cfg["word_segment"] = False

    train, val, test = load_split(args.lang)
    out_dir = Path(cfg["output_dir"]) / args.lang
    best_path = out_dir / "best_params.json"

    best = None
    if args.tune > 0:
        study = tune(cfg, train, val, n_trials=args.tune, study_dir=out_dir,
                     study_name=f"claim_{args.lang}", trainer_cls=ClaimModel)
        best = save_best_params(study, best_path,
                                max_epochs=int(cfg.get("tune", {}).get("max_epochs", 10)))
        print(f"tune xong ({best['n_trials']} trials): val macro-F1 "
              f"{best['val_macro_f1']} | params -> {best_path}")
    elif best_path.exists():
        best = json.loads(best_path.read_text(encoding="utf-8"))
        print(f"dung best params da tune ({best['tuned_at']}, "
              f"val macro-F1 {best['val_macro_f1']})")
    else:
        print("CHUA TUNE — train voi default claim.yml "
              "(nen chay --tune 30 truoc khi lay ket qua chinh)")
    if best:
        cfg = apply_best_params(cfg, best)

    results = {"tuned_params": best,
               "multi_task": multi_seed(cfg, train, val, test, out_dir=out_dir,
                                        trainer_cls=ClaimModel)}

    if args.single_task:
        cb_train = train[train["source"] == "climatebert"]
        results["single_task"] = run_single_task_fallback(cfg, cb_train, val, test)

    metrics_path = Path(f"outputs/metrics/claim_{args.lang}.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["multi_task"]["aggregate"], indent=2))


if __name__ == "__main__":
    main()
