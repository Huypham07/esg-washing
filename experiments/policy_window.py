"""P7: policy-window analysis quanh Thong tu 17/2022 (hieu luc 01/06/2023).

pre = {2020,2021,2022}, post = {2023,2024}. Per-pillar E/S/G, VDR/QDR CORRECTED (acc_correct tren
tung subset, M tu gold). Hai estimator: pooled + mean-of-bank. Bootstrap CI cho delta. Sign test per-bank.
Claim MUC 2 (khong nhan qua, khong DiD). Doc-only tren outputs run B.
  python experiments/policy_window.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from esgwash.indices.correction import build_M, acc_correct  # noqa: E402

PILL = {"E": "is_env", "S": "is_soc", "G": "is_gov"}
PRE, POST = {2020, 2021, 2022}, {2023, 2024}
RNG = np.random.default_rng(42)


def derive(r):
    if int(r["co_so_dinh_luong"]) and int(r["quy_ve_bank"]):
        return 2
    if int(r["co_hanh_dong_ten"]):
        return 1
    return 0


def gold_M(fn):
    m = pd.read_parquet(ROOT / "experiments/eval/gold_classified.parquet")[["chunk_id", "spec_level"]].rename(columns={"spec_level": "m"})
    g = pd.read_excel(ROOT / "data" / fn, sheet_name="Sheet1")
    g = g[g["g_is_commit"] == 1].copy(); g["gold"] = g.apply(derive, axis=1)
    mg = m.merge(g[["chunk_id", "gold"]], on="chunk_id")
    return build_M(mg["m"], mg["gold"])[0]


def rates(levels):
    lv = np.asarray(levels, float)
    return np.array([(lv == 0).mean(), (lv == 1).mean(), (lv == 2).mean()])


def corr_vdr_qdr(levels, M):
    c = acc_correct(rates(levels), M)
    return c[0], c[2]   # VDR, QDR


M = gold_M("gold_annot_1_relabeled.xlsx")           # M_A (primary)

clf = pd.concat([pd.read_parquet(p) for p in sorted((ROOT / "outputs/cti").glob("*/*/classified.parquet"))],
                ignore_index=True)
clf["bank"] = clf["chunk_id"].str.split("_").str[0]
clf["year"] = clf["chunk_id"].str.extract(r"_(\d{4})__")[0].astype(int)
esg = clf[["is_env", "is_soc", "is_gov"]].max(axis=1).astype(bool)
com = clf[(clf["is_commitment"] == 1) & esg].copy()
com["pre"] = com["year"].isin(PRE)

rep = {"split": {"pre": sorted(PRE), "post": sorted(POST)}, "estimator": {}}

# ---------- Pooled estimator + bootstrap CI (resample chunk) ----------
pool = {}
for name, col in PILL.items():
    p = com[(com[col] == 1) & com["pre"]]["spec_level"].to_numpy(float)
    q = com[(com[col] == 1) & ~com["pre"]]["spec_level"].to_numpy(float)
    vp, qp = corr_vdr_qdr(p, M); vq, qq = corr_vdr_qdr(q, M)
    # bootstrap delta CI
    dqs, dvs = [], []
    for _ in range(2000):
        bp = p[RNG.integers(0, len(p), len(p))]; bq = q[RNG.integers(0, len(q), len(q))]
        vpb, qpb = corr_vdr_qdr(bp, M); vqb, qqb = corr_vdr_qdr(bq, M)
        dqs.append(qqb - qpb); dvs.append(vqb - vpb)
    pool[name] = {
        "VDR_pre": round(vp, 3), "VDR_post": round(vq, 3), "dVDR": round(vq - vp, 3),
        "dVDR_CI": [round(np.percentile(dvs, 2.5), 3), round(np.percentile(dvs, 97.5), 3)],
        "QDR_pre": round(qp, 3), "QDR_post": round(qq, 3), "dQDR": round(qq - qp, 3),
        "dQDR_CI": [round(np.percentile(dqs, 2.5), 3), round(np.percentile(dqs, 97.5), 3)],
        "vol_per_report_pre": round(len(p) / (9 * len(PRE)), 1),
        "vol_per_report_post": round(len(q) / (9 * len(POST)), 1)}
rep["estimator"]["pooled"] = pool

# ---------- Mean-of-bank estimator + sign test (QDR per pillar) ----------
mob = {}
for name, col in PILL.items():
    deltas = {}
    for b, g in com[com[col] == 1].groupby("bank"):
        gp = g[g["pre"]]["spec_level"].to_numpy(float); gq = g[~g["pre"]]["spec_level"].to_numpy(float)
        if len(gp) < 5 or len(gq) < 5:
            continue
        deltas[b] = corr_vdr_qdr(gq, M)[1] - corr_vdr_qdr(gp, M)[1]
    vals = np.array(list(deltas.values()))
    up = int((vals > 0).sum())
    st = binomtest(up, len(vals), 0.5)
    # bootstrap CI over banks
    mds = [np.mean(vals[RNG.integers(0, len(vals), len(vals))]) for _ in range(2000)]
    mob[name] = {"mean_dQDR": round(float(vals.mean()), 3),
                 "CI": [round(np.percentile(mds, 2.5), 3), round(np.percentile(mds, 97.5), 3)],
                 "n_banks_up": up, "n_banks": len(vals), "sign_test_p": round(float(st.pvalue), 3),
                 "per_bank": {b: round(float(v), 3) for b, v in sorted(deltas.items(), key=lambda x: -x[1])}}
rep["estimator"]["mean_of_bank"] = mob

# TBR-E pre/post
com["tf"] = com["co_moc_tg"].astype(int)
e = com[com["is_env"] == 1]
rep["TBR_E"] = {"pre": round(float(e[e["pre"]]["tf"].mean()), 3), "post": round(float(e[~e["pre"]]["tf"].mean()), 3)}

print(json.dumps(rep, indent=2, ensure_ascii=False))
(ROOT / "experiments/eval/policy_window_report.json").write_text(
    json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
print("\n-> experiments/eval/policy_window_report.json")
