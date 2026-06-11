"""Baselines: TF-IDF+LR (moi task) + XLM-R zero-shot -> outputs/metrics/baselines.json.

  python scripts/run_baselines.py [--skip-xlmr]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from esgwash.config import load_config
from esgwash.data import claim_merge, topic_merge
from esgwash.models.baselines import tfidf_lr_baseline, xlmr_zero_shot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-xlmr", action="store_true")
    args = ap.parse_args()
    results = {}

    topic_vi = topic_merge.split_stratified(topic_merge.build_masked_table("vi"))
    claim_vi = claim_merge.build_claim_table("vi")
    results["tfidf_lr_topic"] = tfidf_lr_baseline(
        topic_vi[topic_vi["split"] == "train"], topic_vi[topic_vi["split"] == "test"],
        heads=["env", "soc", "gov"])
    results["tfidf_lr_claim"] = tfidf_lr_baseline(
        claim_vi[claim_vi["split"] == "train"], claim_vi[claim_vi["split"] == "test"],
        heads=["commitment", "specificity"])

    if not args.skip_xlmr:
        topic_en = topic_merge.split_stratified(topic_merge.build_masked_table("en"))
        claim_en = claim_merge.build_claim_table("en")
        cfg_t = {**load_config("topic"), "heads": ["env", "soc", "gov"]}
        cfg_c = {**load_config("claim"), "heads": ["commitment", "specificity"]}
        results["xlmr_zero_shot_topic"] = xlmr_zero_shot(
            cfg_t, topic_en[topic_en["split"] == "train"],
            topic_en[topic_en["split"] == "dev"], topic_vi[topic_vi["split"] == "test"])
        results["xlmr_zero_shot_claim"] = xlmr_zero_shot(
            cfg_c, claim_en[claim_en["split"] == "train"],
            claim_en[claim_en["split"] == "dev"], claim_vi[claim_vi["split"] == "test"])

    out = Path("outputs/metrics/baselines.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
