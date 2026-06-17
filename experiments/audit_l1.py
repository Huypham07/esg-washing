"""Audit L1 grounding (Phase 03): phân loại chunk Mức 1 theo support + chẩn loại miss.

zero-support tách 2 loại: n_evidence==0 (anchor/retrieval MISS) vs >0 (có câu nhưng NLI bác).
  python experiments/audit_l1.py [--root outputs/cti_b2] [--bank bidv]
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def best_ev(tr: list) -> dict | None:
    evs = [e for t in tr for e in (t.get("evidence") or [])]
    return max(evs, key=lambda e: e.get("entail", 0) - 0.5 * e.get("contra", 0)) if evs else None


def claims_of(tr: list) -> str:
    return " | ".join(str(t.get("claim", ""))[:50] for t in tr)


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/cti_b2")
    ap.add_argument("--bank", default="bidv")
    a = ap.parse_args(argv)
    for cell in sorted((Path(a.root) / a.bank).glob("*")):
        gp = cell / "claims_grounded.parquet"
        if not gp.exists():
            continue
        clf = pd.read_parquet(cell / "classified.parquet")
        g = pd.read_parquet(gp)
        m = g.merge(clf[["chunk_index", "spec_level"]], on="chunk_index", how="left")
        l1 = m[m["spec_level"] == 1].copy()
        l1["tr"] = l1["item_grounding"].apply(lambda s: json.loads(s) if s else [])
        zero = l1[l1["support"] == 0]
        mid = l1[(l1["support"] > 0) & (l1["support"] < 0.5)]
        hi = l1[l1["support"] >= 0.5]
        nocand = zero[zero["n_evidence"] == 0]
        lowent = zero[zero["n_evidence"] > 0]
        print(f"\n=== {a.bank} {cell.name}: L1={len(l1)} | support=0: {len(zero)} "
              f"(no_candidate:{len(nocand)} low_entail:{len(lowent)}) | mid(0,0.5): {len(mid)} | >=0.5: {len(hi)} ===")
        print("-- zero / NO candidate (anchor/retrieval miss → false-neg tiềm năng) --")
        for _, r in nocand.head(5).iterrows():
            print(f"   claim: {claims_of(r['tr'])[:110]}")
        print("-- zero / có candidate nhưng entail thấp (NLI bác / không corroborate) --")
        for _, r in lowent.head(5).iterrows():
            e = best_ev(r["tr"]) or {}
            print(f"   claim: {claims_of(r['tr'])[:65]} | best entail={e.get('entail')}: {str(e.get('text', ''))[:80]}")
        print("-- mid (0,0.5) --")
        for _, r in mid.head(5).iterrows():
            e = best_ev(r["tr"]) or {}
            print(f"   [sup={r.support:.2f}] {claims_of(r['tr'])[:55]} | entail={e.get('entail')}: {str(e.get('text', ''))[:70]}")


if __name__ == "__main__":
    main()
