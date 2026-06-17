"""Phase 03 — so sánh gCTI B0/B1/B2 với θ PER-LEVEL (θ_L2 cố định, sweep θ_L1).

gCTI cheap = (spec_level==0) | (spec_level==1 & support<θ_L1) | (spec_level==2 & support<θ_L2).
CTI (chỉ Mức 0) giống nhau mọi arm (classify đồng nhất qua reuse). B0/B1: L1 support=0 -> luôn cheap;
B2: L1 support thật. Δ(B2−B1) = tác động THÊM grounding Mức 1; Δ(B1−B0) = tác động ĐỔI model NLI.

  python experiments/compare_arms.py [--theta-l1 0.3 0.4 0.5] [--theta-l2 0.7]
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PILLARS = ("env", "soc", "gov")
ARMS = {"B0": "outputs/cti", "B1": "outputs/cti_b1",
        "B2": "outputs/cti_b2", "B2f": "outputs/cti_b2_floor"}
OUT = Path("experiments/metrics/arms_comparison.json")


def arm_cells(root: str, bank: str, theta_l1: float, theta_l2: float) -> dict:
    """-> {(year, pillar): {n, CTI, gCTI}} cho 1 arm."""
    out = {}
    base = Path(root) / bank
    if not base.exists():
        return out
    for cell in sorted(base.glob("*")):
        gp = cell / "claims_grounded.parquet"
        if not gp.exists():
            continue
        clf = pd.read_parquet(cell / "classified.parquet")
        g = pd.read_parquet(gp)
        commit = clf[clf["is_commitment"] == 1].merge(
            g[["chunk_index", "support"]], on="chunk_index", how="left")
        commit["support"] = commit["support"].fillna(0.0)
        for p in PILLARS:
            c = commit[commit[f"is_{p}"] == 1]
            if not len(c):
                continue
            sl, sup = c["spec_level"], c["support"]
            cheap = (sl == 0) | ((sl == 1) & (sup < theta_l1)) | ((sl == 2) & (sup < theta_l2))
            out[(cell.name, p)] = {"n": int(len(c)), "CTI": round(float((sl == 0).mean()), 3),
                                   "gCTI": round(float(cheap.mean()), 3)}
    return out


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="bidv")
    ap.add_argument("--theta-l2", type=float, default=0.7)
    ap.add_argument("--theta-l1", type=float, nargs="+", default=[0.3, 0.4, 0.5])
    a = ap.parse_args(argv)

    dump = {}
    for tl1 in a.theta_l1:
        arms = {k: arm_cells(r, a.bank, tl1, a.theta_l2) for k, r in ARMS.items()}
        keys = sorted(arms.get("B2") or arms.get("B1") or arms.get("B0") or {})
        print(f"\n##### θ_L1={tl1}  θ_L2={a.theta_l2}  ({a.bank}) #####")
        print(f"{'year-pillar':>14} {'n':>4} {'CTI':>6} {'B0':>6} {'B1':>6} {'B2(.5)':>7} {'B2f(.25)':>9}")
        for k in keys:
            yp = f"{k[0]}-{k[1]}"
            cti = (arms["B2"].get(k) or arms["B1"].get(k) or arms["B0"].get(k) or {}).get("CTI")
            n = (arms["B2"].get(k) or arms["B1"].get(k) or {}).get("n")
            b0 = arms["B0"].get(k, {}).get("gCTI")
            b1 = arms["B1"].get(k, {}).get("gCTI")
            b2 = arms["B2"].get(k, {}).get("gCTI")
            b2f = arms["B2f"].get(k, {}).get("gCTI")
            print(f"{yp:>14} {str(n):>4} {str(cti):>6} {str(b0):>6} {str(b1):>6} {str(b2):>7} {str(b2f):>9}")
        dump[str(tl1)] = {k: {f"{c[0]}-{c[1]}": v for c, v in arms[k].items()} for k in arms}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dump, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
