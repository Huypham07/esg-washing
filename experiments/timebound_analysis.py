"""P04: time-bound dimension (has_timeframe = co_moc_tg) tren panel 45 bao cao.

co_moc_tg KHONG tham gia luat xep muc 0/1/2 -> P(timeframe|level) don dieu = construct validity.
TBR = time-bound rate; TQR = time-bound quantified rate (Muc2 & co deadline); level2-no-deadline = quantified-but-undated.
Doc-only tren outputs run B; KHONG sua derive_flags, KHONG chay lai panel.
  python experiments/timebound_analysis.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]

# --- load panel committed ESG chunks ---
clf = pd.concat([pd.read_parquet(p) for p in sorted((ROOT / "outputs/cti").glob("*/*/classified.parquet"))],
                ignore_index=True)
if "bank" not in clf.columns:
    clf["bank"] = clf["chunk_id"].str.split("_").str[0]
    clf["year"] = clf["chunk_id"].str.extract(r"_(\d{4})__")[0].astype(int)
esg = clf[["is_env", "is_soc", "is_gov"]].max(axis=1).astype(bool)
com = clf[(clf["is_commitment"] == 1) & esg].copy()
com["tf"] = com["co_moc_tg"].astype(int)
com["lv"] = com["spec_level"].astype(int)
n = len(com)


def mean_of_cells(sub, col="tf"):
    return float(sub.groupby(["bank", "year"])[col].mean().mean())


rep = {"n_committed": int(n)}

# 1. TBR overall (mean-of-cells + pooled)
rep["TBR"] = {"mean_of_cells": round(mean_of_cells(com), 4),
              "pooled": round(float(com["tf"].mean()), 4)}

# 2. TBR by pillar (pooled among pillar-tagged committed)
rep["TBR_by_pillar"] = {}
for name, colp in {"E": "is_env", "S": "is_soc", "G": "is_gov"}.items():
    sub = com[com[colp] == 1]
    rep["TBR_by_pillar"][name] = round(float(sub["tf"].mean()), 4)

# 3. TBR by year (mean-of-cells) + Spearman trend
by_year = com.groupby(["bank", "year"])["tf"].mean().reset_index()
yr = by_year.groupby("year")["tf"].mean()
rep["TBR_by_year"] = {int(y): round(float(v), 4) for y, v in yr.items()}
rho, p = spearmanr(by_year["year"], by_year["tf"])
rep["TBR_year_spearman"] = {"rho": round(float(rho), 4), "p": float(f"{p:.4g}")}

# 4. P(timeframe | level) — pooled conditional (kiem don dieu)
rep["P_timeframe_given_level"] = {}
for k in (0, 1, 2):
    sub = com[com["lv"] == k]
    rep["P_timeframe_given_level"][k] = round(float(sub["tf"].mean()), 4) if len(sub) else None
g = rep["P_timeframe_given_level"]
rep["monotonic_increasing"] = bool(g[0] < g[1] < g[2])

# 5. level2-no-deadline (quantified-but-undated) = P(no timeframe | level 2)
lvl2 = com[com["lv"] == 2]
rep["level2_no_deadline"] = {"overall": round(float((lvl2["tf"] == 0).mean()), 4),
                             "n_level2": int(len(lvl2))}
rep["level2_no_deadline"]["by_pillar"] = {}
for name, colp in {"E": "is_env", "S": "is_soc", "G": "is_gov"}.items():
    s2 = lvl2[lvl2[colp] == 1]
    rep["level2_no_deadline"]["by_pillar"][name] = round(float((s2["tf"] == 0).mean()), 4) if len(s2) else None

# 6. TQR = time-bound quantified rate = share(level==2 & timeframe) among committed
com["tqr"] = ((com["lv"] == 2) & (com["tf"] == 1)).astype(int)
rep["TQR"] = {"mean_of_cells": round(mean_of_cells(com, "tqr"), 4),
              "pooled": round(float(com["tqr"].mean()), 4)}

print(json.dumps(rep, indent=2, ensure_ascii=False))
(ROOT / "experiments/eval/timebound_summary.json").write_text(
    json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
print("\n-> experiments/eval/timebound_summary.json")
