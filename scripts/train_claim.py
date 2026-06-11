"""Train M2 claim (multi-task commitment + specificity) theo configs/claim.yml.

  python scripts/train_claim.py [--lang vi|en] [--quick] [--single-task]
--single-task: chay them fallback 2 model rieng cho ablation.
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

GOLD = Path("data/processed/gold/claim_table.parquet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="vi", choices=["vi", "en"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--single-task", action="store_true")
    args = ap.parse_args()

    cfg = load_config("claim")
    if args.quick:
        cfg["train"]["seeds"] = cfg["train"]["seeds"][:1]

    if args.lang == "vi" and GOLD.exists():
        df = pd.read_parquet(GOLD)
    else:
        df = claim_merge.build_claim_table(
            lang=args.lang,
            augment_action=cfg.get("augment_action_500", True),
            aux_env_claims=cfg.get("aux_head_env_claims", True),
            augment_ml_promise=cfg.get("augment_ml_promise", False))
    if args.lang == "en":
        cfg["backbone"] = cfg.get("en_backbone", "roberta-base")
        cfg["word_segment"] = False

    heads = list(MAIN_HEADS) + (["claim"] if cfg.get("aux_head_env_claims") else [])
    cfg["heads"] = heads
    train = df[df["split"] == "train"]
    dev = df[(df["split"] == "dev") & (df["source"] == "climatebert")]
    test = df[df["split"] == "test"]
    out_dir = Path(cfg["output_dir"]) / args.lang
    results = {"multi_task": multi_seed(cfg, train, dev, test, out_dir=out_dir,
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
