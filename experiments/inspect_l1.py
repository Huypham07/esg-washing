"""Inspect L1 grounding output (Phase 02): phân bố support_L1 + trace audit cho chunk Mức 1.

  python experiments/inspect_l1.py [--root outputs/cti_b2] [--bank bidv]
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def dist(d: pd.DataFrame, lbl: str) -> None:
    s = d["support"]
    if not len(d):
        print(f"  {lbl}: (rỗng)")
        return
    print(f"  {lbl}: n={len(d)} support>0={int((s > 0).sum())} "
          f"median={s.median():.3f} p75={s.quantile(.75):.3f} "
          f">=0.3={(s >= 0.3).mean():.2f} >=0.5={(s >= 0.5).mean():.2f} >=0.7={(s >= 0.7).mean():.2f}")


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/cti_b2")
    ap.add_argument("--bank", default="bidv")
    a = ap.parse_args(argv)
    root = Path(a.root) / a.bank
    for cell in sorted(root.glob("*")):
        g_path = cell / "claims_grounded.parquet"
        if not g_path.exists():
            continue
        clf = pd.read_parquet(cell / "classified.parquet")
        g = pd.read_parquet(g_path)
        m = g.merge(clf[["chunk_index", "spec_level"]], on="chunk_index", how="left")
        l1, l2 = m[m["spec_level"] == 1], m[m["spec_level"] == 2]
        print(f"=== {a.bank} {cell.name} ===")
        dist(l1, "L1"); dist(l2, "L2")
        for _, r in l1.sort_values("support", ascending=False).head(5).iterrows():
            tr = json.loads(r["item_grounding"]) if r["item_grounding"] else []
            best = max(tr, key=lambda t: t.get("support", 0)) if tr else {}
            evs = best.get("evidence") or []
            # câu DRIVING support = max(entail - 0.5*contra), không phải sim-top
            ev = max(evs, key=lambda e: e.get("entail", 0) - 0.5 * e.get("contra", 0)) if evs else {}
            print(f"  [sup={r.support:.3f}] claim: {str(best.get('claim', ''))[:90]}")
            print(f"      ev(entail={ev.get('entail')}, contra={ev.get('contra')}, "
                  f"chunk={ev.get('src_chunk')}): {str(ev.get('text', ''))[:120]}")


if __name__ == "__main__":
    main()
