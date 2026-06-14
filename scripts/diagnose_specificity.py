"""Diagnostic chen-so (spec 05 V2): so flip-rate cua cac specificity scorer.

  python scripts/diagnose_specificity.py [--n 40] [--scorers digit,llm]

Seed = cau non-specific (gold specificity label=0, dich VI) + vai cau curated ngan hang.
DigitPresenceScorer (shortcut hien ngon) ky vong flip ~1; SpecificityLLM (Qwen3-0.6B,
rubric) ky vong flip ~0. Ket qua -> outputs/metrics/specificity_diagnostic.json.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from esgwash.config import load_config
from esgwash.validation.digit_shortcut import (CURATED_VAGUE, DigitPresenceScorer,
                                               PERTURBATIONS, flip_rates)

GOLD = Path("data/translate/specificity.test.parquet")


def load_seeds(n: int) -> list[str]:
    seeds = list(CURATED_VAGUE)
    if GOLD.exists():
        df = pd.read_parquet(GOLD)
        neg = df[df["label"] == 0]["text"].astype(str).tolist()
        seeds += neg
    return seeds[:n]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="so seed (cau non-specific)")
    ap.add_argument("--scorers", default="digit,llm", help="digit,llm (phay ngan cach)")
    args = ap.parse_args(argv)

    seeds = load_seeds(args.n)
    want = [s.strip() for s in args.scorers.split(",")]
    scorers = {}
    if "digit" in want:
        scorers["digit_presence"] = DigitPresenceScorer()
    if "llm" in want:
        from esgwash.models.specificity_llm import SpecificityLLM
        scorers["specificity_llm"] = SpecificityLLM(load_config("specificity"))

    results = {"n_seed": len(seeds), "perturbations": list(PERTURBATIONS),
               "scorers": {}}
    for name, sc in scorers.items():
        print(f"[{name}] dang cham {len(seeds)} seed x {len(PERTURBATIONS)} phep chen...")
        results["scorers"][name] = flip_rates(sc, seeds)
        print(json.dumps(results["scorers"][name], indent=2, ensure_ascii=False))

    out = Path("outputs/metrics/specificity_diagnostic.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
