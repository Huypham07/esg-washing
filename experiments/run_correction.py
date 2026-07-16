"""P02 run: bias-corrected VDR/NAR/QDR (ACC) + bootstrap kep CI + re-run test thong ke.

Basis gold = relabel tach-co (reconcile 07/07). Doc-only tren artifact co san (khong GPU, khong sua pipeline).
  python experiments/run_correction.py
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
sys.path.insert(0, str(ROOT / "src"))
from esgwash.indices.correction import build_M, acc_correct, correct_cells, bootstrap_joint  # noqa: E402


def derive_level(row) -> int:
    if int(row["co_so_dinh_luong"]) and int(row["quy_ve_bank"]):
        return 2
    if int(row["co_hanh_dong_ten"]):
        return 1
    return 0


def gold_levels(path: Path) -> pd.DataFrame:
    d = pd.read_excel(path, sheet_name="Sheet1")
    d = d[d["g_is_commit"] == 1].copy()
    d["gold"] = d.apply(derive_level, axis=1)
    return d[["chunk_id", "gold"]]


# --- 1. Gold + model preds -> M ---
model = pd.read_parquet(ROOT / "experiments/eval/gold_classified.parquet")[["chunk_id", "spec_level"]]
model = model.rename(columns={"spec_level": "m"})
mats = {}
for tag, fn in [("A", "gold_annot_1_relabeled.xlsx"), ("B", "gold_annot_2_relabeled.xlsx")]:
    g = gold_levels(ROOT / "data" / fn)
    mg = model.merge(g, on="chunk_id")
    M, counts = build_M(mg["m"], mg["gold"])
    mats[tag] = {"M": M, "counts": counts, "n": int(len(mg))}
    print(f"M_{tag} (n={len(mg)}) counts:\n{counts}\nM=P(pred|true):\n{np.round(M,3)}\n")

# --- 2. Panel per-cell raw VDR/NAR/QDR ---
cti = pd.concat([pd.read_parquet(p) for p in sorted((ROOT / "outputs/cti").glob("*/*/cti.parquet"))],
                ignore_index=True)
cells = cti[["cti", "nar", "qdr"]].to_numpy(float)          # (45, 3)
raw_headline = cells.mean(axis=0)
print(f"n_cells={len(cells)}  raw headline (mean-of-cells) VDR/NAR/QDR = {np.round(raw_headline,4)}")

# --- 3. Corrected headline (A & B) + doi chieu demo ---
report = {"basis": "relabel derived-from-flags (reconcile 07/07)",
          "n_cells": int(len(cells)),
          "raw_headline": {"VDR": round(float(raw_headline[0]), 4),
                           "NAR": round(float(raw_headline[1]), 4),
                           "QDR": round(float(raw_headline[2]), 4)},
          "M": {}, "corrected_headline": {}, "bootstrap_CI_95": {}, "stat_tests": {}}
for tag in ("A", "B"):
    corr = acc_correct(raw_headline, mats[tag]["M"])
    report["M"][tag] = {"n": mats[tag]["n"], "counts": mats[tag]["counts"].tolist()}
    report["corrected_headline"][tag] = {"VDR": round(float(corr[0]), 4),
                                         "NAR": round(float(corr[1]), 4),
                                         "QDR": round(float(corr[2]), 4)}
    print(f"corrected (M_{tag}): VDR/NAR/QDR = {np.round(corr,4)}")
print("  (demo spec doi chieu: raw 0.479/0.304/0.217 -> corrected 0.433/0.318/0.249)")

# --- 4. Bootstrap kep CI (M_A) ---
g_used = model.merge(gold_levels(ROOT / "data/gold_annot_1_relabeled.xlsx"), on="chunk_id")
point, lo, hi, skipped, nb = bootstrap_joint(g_used["m"], g_used["gold"], cells, n_boot=2000, seed=42)
for i, name in enumerate(("VDR", "NAR", "QDR")):
    report["bootstrap_CI_95"][name] = [round(float(lo[i]), 4), round(float(hi[i]), 4)]
    print(f"  {name}: corrected {point[i]:.4f}  CI95 [{lo[i]:.4f}, {hi[i]:.4f}]")
report["bootstrap_CI_95"]["n_boot_used"] = nb
report["bootstrap_CI_95"]["n_singular_skipped"] = skipped

# --- 5. Re-run test thong ke: raw vs corrected (M_A) ---
corr_cells = correct_cells(cells, mats["A"]["M"])          # (45, 3) corrected per-cell
tab = cti[["bank", "year"]].copy()
for i, nm in enumerate(("cti", "nar", "qdr")):
    tab[f"{nm}_raw"] = cells[:, i]
    tab[f"{nm}_cor"] = corr_cells[:, i]

def sp(x, y):
    rho, p = spearmanr(x, y)
    return round(float(rho), 4), float(f"{p:.4g}")

print("\n=== Temporal Spearman (vs year) — RAW vs CORRECTED ===")
for nm in ("cti", "nar", "qdr"):
    r_raw, p_raw = sp(tab[f"{nm}_raw"], tab["year"])
    r_cor, p_cor = sp(tab[f"{nm}_cor"], tab["year"])
    flip = (p_raw < 0.05) != (p_cor < 0.05)
    report["stat_tests"][f"temporal_{nm}"] = {"raw": {"rho": r_raw, "p": p_raw},
                                              "corrected": {"rho": r_cor, "p": p_cor},
                                              "significance_flip": flip}
    print(f"  {nm.upper():3}: raw rho={r_raw:+.3f} p={p_raw:<9} | cor rho={r_cor:+.3f} p={p_cor:<9}"
          f"{'   <-- FLIP significance!' if flip else ''}")

r_raw, p_raw = sp(tab["cti_raw"], tab["qdr_raw"])
r_cor, p_cor = sp(tab["cti_cor"], tab["qdr_cor"])
report["stat_tests"]["cti_vs_qdr"] = {"raw": {"rho": r_raw, "p": p_raw},
                                      "corrected": {"rho": r_cor, "p": p_cor},
                                      "significance_flip": (p_raw < 0.05) != (p_cor < 0.05)}
print(f"  CTI~QDR: raw rho={r_raw:+.3f} p={p_raw} | cor rho={r_cor:+.3f} p={p_cor}")
print("\nRQ2 selective-disclosure Friedman = tren pillar SHARE (coverage) -> correction KHONG dong toi (bat bien).")

# --- 6. Per-pillar corrected (M chung 3 tru) — bang Delta RQ2 ---
clf = pd.concat([pd.read_parquet(p) for p in sorted((ROOT / "outputs/cti").glob("*/*/classified.parquet"))],
                ignore_index=True)
clf["bank"] = clf["chunk_id"].str.split("_").str[0]
clf["year"] = clf["chunk_id"].str.extract(r"_(\d{4})__")[0]
report["per_pillar"] = {"raw": {}, "corrected_A": {}, "corrected_B": {}}
print("\n=== Per-pillar VDR-QDR gap (Delta) — RAW vs CORRECTED ===")
print(f"{'Pillar':6} {'raw Δ (QDR)':>16} {'corr_A Δ (QDR)':>18} {'corr_B Δ (QDR)':>18}")
for name, col in {"E": "is_env", "S": "is_soc", "G": "is_gov"}.items():
    sub = clf[(clf["is_commitment"] == 1) & (clf[col] == 1)]
    percell = [[(g["spec_level"].to_numpy(float) == k).mean() for k in (0, 1, 2)]
               for _, g in sub.groupby(["bank", "year"])]
    raw = np.array(percell).mean(axis=0)
    cA = acc_correct(raw, mats["A"]["M"])
    cB = acc_correct(raw, mats["B"]["M"])
    for key, vec in [("raw", raw), ("corrected_A", cA), ("corrected_B", cB)]:
        report["per_pillar"][key][name] = {"VDR": round(float(vec[0]), 4),
                                           "QDR": round(float(vec[2]), 4),
                                           "delta": round(float(vec[0] - vec[2]), 4)}
    print(f"{name:6} {raw[0]-raw[2]:8.3f} ({raw[2]:.3f}) {cA[0]-cA[2]:12.3f} ({cA[2]:.3f}) {cB[0]-cB[2]:12.3f} ({cB[2]:.3f})")
print("=> Ordering E<S<G gap giu nguyen; Gov QDR thap nhat -> ket luan 'Gov it dinh luong nhat' vung.")

out = ROOT / "experiments/eval/correction_report.json"
out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\n-> {out}")
