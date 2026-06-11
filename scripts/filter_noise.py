"""QE filter cho gold dich (spec 01 #4): cham diem (EN, VI) -> loc day phan phoi.

  python scripts/filter_noise.py [--backend labse|cometkiwi] [--q 0.05]
Ghi diem QE vao data/processed/gold/*_qe.parquet (giu nguyen bang goc,
train script loc theo cot qe_score khi chay ablation).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from esgwash.data.noise_filter import qe_scores

GOLD = Path("data/processed/gold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="labse", choices=["labse", "cometkiwi"])
    ap.add_argument("--q", type=float, default=0.05)
    args = ap.parse_args()

    for name in ("topic_masked", "claim_table"):
        df = pd.read_parquet(GOLD / f"{name}.parquet")
        has_pair = df["text_en"].notna() & df["text"].notna()
        scores = np.full(len(df), np.nan)
        scores[has_pair.values] = qe_scores(df.loc[has_pair, "text_en"].tolist(),
                                            df.loc[has_pair, "text"].tolist(),
                                            backend=args.backend)
        df["qe_score"] = scores
        cutoff = np.nanquantile(scores, args.q)
        df["qe_keep"] = df["qe_score"].isna() | (df["qe_score"] > cutoff)
        df.to_parquet(GOLD / f"{name}_qe.parquet", index=False)
        print(f"{name}: cutoff(q={args.q})={cutoff:.4f}, "
              f"loc {(~df['qe_keep']).sum()}/{len(df)} dong")


if __name__ == "__main__":
    main()
