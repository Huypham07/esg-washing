"""Review-audit splits 2 track (gold/silver): size, pos-rate, leakage, domain-overlap.

Chạy:  python outputs/_review_data_audit.py
Mục đích: kiểm tra TÍNH TOÀN VẸN trước Phase 04 (không train gì).
"""
import os
import sys
import unicodedata
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")  # bài học cũ: print VN crash cp1252
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

TASKS = ["env", "soc", "gov", "commitment", "specificity"]
SPLITS = ["train", "val", "test"]
TRACKS = {"gold": "data/vi_gold", "silver": "data/vi_silver"}


def textcol(df: pd.DataFrame) -> str:
    for c in ("sentence", "text"):
        if c in df.columns:
            return c
    return df.columns[0]


def load(track_dir: str, task: str, split: str):
    p = Path(track_dir) / task / f"{split}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def norm_set(s: pd.Series) -> set:
    return set(s.map(lambda x: unicodedata.normalize("NFC", str(x)).strip()))


store = {}
for track, d in TRACKS.items():
    for task in TASKS:
        for split in SPLITS:
            store[(track, task, split)] = load(d, task, split)

print("=" * 72)
print("PART 0 — schema (cols/dtypes) mẫu gold/env/train & silver/env/train")
print("=" * 72)
for track in ("gold", "silver"):
    df = store[(track, "env", "train")]
    if df is not None:
        print(f"  {track}/env/train cols={list(df.columns)} dtypes={dict(df.dtypes.astype(str))}")

print("\n" + "=" * 72)
print("PART 1 — SIZE + LABEL DIST per split (pos = label==1)")
print("=" * 72)
for track, d in TRACKS.items():
    print(f"\n### TRACK={track}  ({d})")
    for task in TASKS:
        cells = []
        for split in SPLITS:
            df = store[(track, task, split)]
            if df is None:
                cells.append(f"{split}: --")
                continue
            lab = df["label"] if "label" in df.columns else df.iloc[:, -1]
            n = len(df)
            pos = int((lab == 1).sum())
            cells.append(f"{split} n={n} pos={pos} ({pos/n:.0%})" if n else f"{split} empty")
        print(f"  {task:12s} | " + " | ".join(cells))

print("\n" + "=" * 72)
print("PART 2 — LEAKAGE within task (overlap câu giữa splits; kỳ vọng 0)")
print("=" * 72)
for track in TRACKS:
    print(f"\n### TRACK={track}")
    for task in TASKS:
        sets = {}
        for split in SPLITS:
            df = store[(track, task, split)]
            if df is not None:
                sets[split] = norm_set(df[textcol(df)])
        if len(sets) < 3:
            print(f"  {task:12s} | incomplete ({list(sets)})")
            continue
        tv = len(sets["train"] & sets["val"])
        tt = len(sets["train"] & sets["test"])
        vt = len(sets["val"] & sets["test"])
        flag = " <-- LEAK!" if (tv or tt or vt) else " ok"
        print(f"  {task:12s} | train∩val={tv} train∩test={tt} val∩test={vt}{flag}")

print("\n" + "=" * 72)
print("PART 3 — DOMAIN separation: gold_test vs silver_test (overlap câu, mỗi task)")
print("=" * 72)
for task in TASKS:
    g = store[("gold", task, "test")]
    s = store[("silver", task, "test")]
    if g is None or s is None:
        print(f"  {task:12s} | missing")
        continue
    gs, ss = norm_set(g[textcol(g)]), norm_set(s[textcol(s)])
    print(f"  {task:12s} | gold_test={len(gs)} silver_test={len(ss)} overlap={len(gs & ss)}")

print("\n" + "=" * 72)
print("PART 4 — FULL cross-track overlap (gold mọi-split vs silver mọi-split)")
print("=" * 72)
for task in TASKS:
    gall, sall = set(), set()
    for split in SPLITS:
        g = store[("gold", task, split)]
        s = store[("silver", task, split)]
        if g is not None:
            gall |= norm_set(g[textcol(g)])
        if s is not None:
            sall |= norm_set(s[textcol(s)])
    print(f"  {task:12s} | gold_all={len(gall)} silver_all={len(sall)} overlap={len(gall & sall)}")

print("\nDONE.")
