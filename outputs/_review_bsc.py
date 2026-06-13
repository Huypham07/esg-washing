"""Soi bsc: vì sao CTI cao nhất. Đọc enriched_corpus_silver.parquet."""
import os
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
os.chdir(Path(__file__).resolve().parents[1])

df = pd.read_parquet("outputs/index/enriched_corpus_silver.parquet")

print("=== Quy mô mỗi bank (tổng câu) ===")
print(df.groupby("bank").size().sort_values().to_string())

b = df[df.bank == "bsc"]
com = b[b.commitment == 1]
vague = com[com.specificity == 0]
print(f"\n=== bsc: tổng {len(b)} câu | ESG {int(b.is_esg.sum())} | commit {len(com)} "
      f"| vague(spec=0) {len(vague)} | specific {int((com.specificity == 1).sum())} "
      f"| CTI {len(vague)/len(com):.2f} ===")

print("\n--- bsc theo NĂM ---")
for y, g in b.groupby("year"):
    c = g[g.commitment == 1]
    v = c[c.specificity == 0]
    line = f"  {y}: câu={len(g)} esg={int(g.is_esg.sum())} commit={len(c)}"
    line += f" vague={len(v)} CTI={len(v)/len(c):.2f}" if len(c) else " commit=0"
    print(line)

print("\n--- bsc theo PILLAR ---")
for p in ["E", "S", "G"]:
    sub = b[b[f"is_{p}"] == 1]
    c = sub[sub.commitment == 1]
    v = c[c.specificity == 0]
    line = f"  {p}: câu={len(sub)} commit={len(c)}"
    line += f" vague={len(v)} CTI={len(v)/len(c):.2f}" if len(c) else " commit=0"
    print(line)

print("\n--- VÍ DỤ cam kết MƠ HỒ (commit=1, spec=0) — đẩy CTI cao ---")
for s in vague.sentence.head(10):
    print("  •", str(s)[:220])

print("\n--- VÍ DỤ cam kết CỤ THỂ (commit=1, spec=1) — đối chứng ---")
for s in com[com.specificity == 1].sentence.head(5):
    print("  •", str(s)[:220])
