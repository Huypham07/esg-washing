"""Xuất CSV để KIỂM TRA TAY grounding Mức 1 (commitment + bằng chứng).

Mỗi dòng = một hành động Mức 1: văn bản cam kết, hành động trích ra, điểm support, câu bằng chứng
tốt nhất (kèm entail/contra/sim), và kết luận (thực chất / cheap theo θ_L1=0.4).
  python experiments/export_l1_review.py [--root outputs/cti_b2] [--bank bidv]
-> experiments/analyse_bidv/l1_review_<root>.csv  (utf-8-sig — mở Excel đọc tiếng Việt OK)
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

THETA_L1 = 0.4
PILLARS = ("env", "soc", "gov")


def best_ev(evs: list) -> dict:
    return max(evs, key=lambda e: e.get("entail", 0) - 0.5 * e.get("contra", 0)) if evs else {}


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/cti_b2")
    ap.add_argument("--bank", default="bidv")
    ap.add_argument("--years", nargs="+", type=int, default=[2023, 2024])
    a = ap.parse_args(argv)

    rows = []
    for yr in a.years:
        d = Path(a.root) / a.bank / str(yr)
        clf = pd.read_parquet(d / "classified.parquet")
        gi = pd.read_parquet(d / "claims_grounded.parquet").set_index("chunk_index")
        # CHỈ ESG (≥1 trụ E/S/G) — khớp mẫu số gCTI per-trụ; loại commitment non-ESG
        m1 = clf[(clf.is_commitment == 1) & (clf.spec_level == 1)
                 & (clf[[f"is_{p}" for p in PILLARS]].sum(axis=1) > 0)]
        for _, c in m1.iterrows():
            ci = int(c.chunk_index)
            raw = gi.loc[ci, "item_grounding"] if ci in gi.index else None
            tr = json.loads(raw) if raw else []
            chunk_sup = round(float(gi.loc[ci, "support"]), 3) if ci in gi.index else 0.0
            decision = "thực chất" if chunk_sup >= THETA_L1 else "cheap (nói suông)"
            pillar = "/".join(p for p in PILLARS if c[f"is_{p}"] == 1) or "(non-ESG)"
            base = {"nam": yr, "chunk": ci, "tru": pillar, "chunk_support": chunk_sup,
                    "ket_luan_chunk": decision,
                    "van_ban_cam_ket": str(c.content_text)[:300].replace("\n", " ")}
            if not tr:
                rows.append({**base, "hanh_dong": "(không trích được action)", "item_support": None,
                             "bang_chung": "", "entail": None, "contra": None, "sim": None,
                             "cau_nguon": None, "so_cau_xet": 0})
                continue
            for it in tr:
                ev = best_ev(it.get("evidence") or [])
                rows.append({**base, "hanh_dong": it.get("claim", ""),
                             "item_support": round(float(it.get("support", 0)), 3),
                             "bang_chung": str(ev.get("text", "")),
                             "entail": ev.get("entail"), "contra": ev.get("contra"),
                             "sim": ev.get("sim"), "cau_nguon": ev.get("src_chunk"),
                             "so_cau_xet": len(it.get("evidence") or [])})

    cols = ["nam", "chunk", "tru", "hanh_dong", "item_support", "chunk_support", "ket_luan_chunk",
            "bang_chung", "entail", "contra", "sim", "cau_nguon", "so_cau_xet", "van_ban_cam_ket"]
    df = pd.DataFrame(rows)[cols]
    out = Path("experiments/analyse_bidv") / f"l1_review_{Path(a.root).name}"
    df.to_csv(out.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    df.to_parquet(out.with_suffix(".parquet"), index=False)
    print(f"-> {out}.csv + {out}.parquet  ({len(df)} dòng hành động Mức 1; floor={Path(a.root).name})")
    print(df.groupby(["nam", "ket_luan_chunk"]).size().to_string())


if __name__ == "__main__":
    main()
