"""Phan tich panel 6 ngan hang x 5 nam (2020-2024) cho RQ1-RQ3 + validation thong ke.

Doc outputs/cti/*/*/ (cti, pillar_shares, info_check, classified) -> experiments/panel/:
  panel.csv      — CTI/NAR/QDR + n_commit moi (bank, year)
  summary.json   — thong ke mo ta, kiem dinh, on dinh ranking
  report.md      — dien giai RQ1 (prevalence) / RQ2 (selective disclosure) / RQ3 (substance gap)

Chi dung output da co (khong GPU): bootstrap CI, Friedman/Wilcoxon, Spearman trend, Kendall tau.
"""
from __future__ import annotations

import glob
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, spearmanr, wilcoxon

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from esgwash.indices.bootstrap import bootstrap_ci  # noqa: E402
from esgwash.validation.sensitivity import (  # noqa: E402
    bootstrap_ranking_stability,
    leave_one_year_out,
)

ROOT = Path("outputs/cti")
OUT = Path("experiments/panel")
PILLARS = ["env", "soc", "gov"]
INDICES = ["cti", "nar", "qdr"]


def _read_glob(pattern: str) -> pd.DataFrame:
    frames = [pd.read_parquet(f) for f in sorted(glob.glob(pattern))]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_panel() -> dict:
    cti = _read_glob(str(ROOT / "*/*/cti.parquet"))
    shares = _read_glob(str(ROOT / "*/*/pillar_shares.parquet"))
    classified = _read_glob(str(ROOT / "*/*/classified.parquet"))
    info = pd.DataFrame([json.loads(Path(f).read_text(encoding="utf-8"))
                         for f in sorted(glob.glob(str(ROOT / "*/*/info_check.json")))])
    return {"cti": cti, "shares": shares, "classified": classified, "info": info}


def descriptive(cti: pd.DataFrame) -> dict:
    """RQ1: mean +/- sd CTI/NAR/QDR overall, theo nam, theo ngan hang + bootstrap CI overall."""
    out = {"n_panels": int(len(cti)), "n_banks": int(cti["bank"].nunique()),
           "years": sorted(int(y) for y in cti["year"].unique())}
    for idx in INDICES:
        point, lo, hi = bootstrap_ci(cti[idx].to_numpy(), np.mean, 2000, 0.95)
        out[f"{idx}_overall"] = {"mean": round(point, 4), "sd": round(float(cti[idx].std()), 4),
                                 "ci95": [round(lo, 4), round(hi, 4)]}
    out["by_year"] = (cti.groupby("year")[INDICES + ["n_commit"]].mean().round(4)
                      .reset_index().to_dict("records"))
    out["by_bank"] = (cti.groupby("bank")[INDICES + ["n_commit"]].mean().round(4)
                      .reset_index().to_dict("records"))
    return out


def selective_disclosure(shares: pd.DataFrame) -> dict:
    """RQ2: tru nao bi ne? Friedman tren share E/S/G theo block (bank, year) + Wilcoxon hau kiem."""
    wide = shares.pivot_table(index=["bank", "year"], columns="pillar",
                              values="share").dropna()
    stat, p = friedmanchisquare(wide["env"], wide["soc"], wide["gov"])
    out = {"n_blocks": int(len(wide)),
           "pillar_mean_share": {p_: round(float(wide[p_].mean()), 4) for p_ in PILLARS},
           "friedman": {"chi2": round(float(stat), 4), "p": float(f"{p:.3e}")}}
    # hau kiem cap doi (env la tru nghi bi ne)
    out["wilcoxon"] = {}
    for a, b in [("env", "gov"), ("env", "soc"), ("soc", "gov")]:
        w, pw = wilcoxon(wide[a], wide[b])
        out["wilcoxon"][f"{a}_vs_{b}"] = {"stat": round(float(w), 2), "p": float(f"{pw:.3e}"),
                                          "median_diff": round(float((wide[a] - wide[b]).median()), 4)}
    return out


def temporal_trend(cti: pd.DataFrame) -> dict:
    """RQ3: CTI/QDR co xu huong theo nam? Spearman pooled + so n_commit dau/cuoi."""
    out = {}
    for idx in INDICES:
        rho, p = spearmanr(cti["year"], cti[idx])
        out[idx] = {"spearman_rho": round(float(rho), 4), "p": float(f"{p:.3e}")}
    rho_n, p_n = spearmanr(cti["year"], cti["n_commit"])
    out["n_commit"] = {"spearman_rho": round(float(rho_n), 4), "p": float(f"{p_n:.3e}")}
    return out


def component_corr(cti: pd.DataFrame) -> dict:
    """Tuong quan CTI voi khoi luong cam ket va QDR (mo ta noi tai chi so)."""
    pairs = [("cti", "qdr"), ("cti", "n_commit"), ("qdr", "n_commit")]
    out = {}
    for a, b in pairs:
        rho, p = spearmanr(cti[a], cti[b])
        out[f"{a}_vs_{b}"] = {"spearman_rho": round(float(rho), 4), "p": float(f"{p:.3e}")}
    return out


def md_block(title: str, rows: list[dict], cols: list[str]) -> str:
    head = "| " + " | ".join(cols) + " |\n| " + " | ".join(["---"] * len(cols)) + " |\n"
    body = "".join("| " + " | ".join(f"{r[c]}" for c in cols) + " |\n" for r in rows)
    return f"\n**{title}**\n\n{head}{body}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    P = load_panel()
    cti = P["cti"].sort_values(["bank", "year"]).reset_index(drop=True)
    cti.to_csv(OUT / "panel.csv", index=False)

    summary = {
        "descriptive": descriptive(cti),
        "selective_disclosure": selective_disclosure(P["shares"]),
        "temporal_trend": temporal_trend(cti),
        "component_corr": component_corr(cti),
        "ranking_stability": bootstrap_ranking_stability(P["classified"], 2000),
        "leave_one_year_out": leave_one_year_out(P["classified"]),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                      encoding="utf-8")

    d = summary["descriptive"]
    sd = summary["selective_disclosure"]
    tr = summary["temporal_trend"]
    rs = summary["ranking_stability"]
    R = [f"# Panel ESG-washing — {d['n_banks']} ngan hang x {len(d['years'])} nam "
         f"({d['n_panels']} panel)", ""]
    R += ["## RQ1 — Prevalence (CTI/NAR/QDR)",
          f"- CTI = {d['cti_overall']['mean']} (sd {d['cti_overall']['sd']}, "
          f"CI95 {d['cti_overall']['ci95']}) · NAR = {d['nar_overall']['mean']} · "
          f"QDR = {d['qdr_overall']['mean']}"]
    R += [md_block("Theo nam", d["by_year"], ["year"] + INDICES + ["n_commit"])]
    R += [md_block("Theo ngan hang", d["by_bank"], ["bank"] + INDICES + ["n_commit"])]

    R += ["\n## RQ2 — Selective disclosure",
          f"- Share trung binh: E={sd['pillar_mean_share']['env']} · "
          f"S={sd['pillar_mean_share']['soc']} · G={sd['pillar_mean_share']['gov']}",
          f"- Friedman chi2={sd['friedman']['chi2']}, p={sd['friedman']['p']} "
          f"(n_block={sd['n_blocks']})",
          f"- Wilcoxon env_vs_gov: p={sd['wilcoxon']['env_vs_gov']['p']}, "
          f"median_diff={sd['wilcoxon']['env_vs_gov']['median_diff']}"]

    R += ["\n## RQ3 — Substance gap / xu huong",
          f"- CTI~year Spearman rho={tr['cti']['spearman_rho']} (p={tr['cti']['p']})",
          f"- QDR~year Spearman rho={tr['qdr']['spearman_rho']} (p={tr['qdr']['p']})",
          f"- n_commit~year rho={tr['n_commit']['spearman_rho']} (p={tr['n_commit']['p']})"]

    R += ["\n## Validation — on dinh ranking (sensitivity)",
          f"- Bootstrap Kendall tau mean={rs['kendall_tau_mean']} "
          f"(p05={rs['kendall_tau_p05']}), top-1 retention={rs['top1_retention']}",
          f"- Leave-one-year-out tau_min={summary['leave_one_year_out']['tau_min']}"]
    (OUT / "report.md").write_text("\n".join(R), encoding="utf-8")
    print(f"-> {OUT}/ : panel.csv, summary.json, report.md")
    print(f"  CTI={d['cti_overall']['mean']} NAR={d['nar_overall']['mean']} "
          f"QDR={d['qdr_overall']['mean']} | Friedman p={sd['friedman']['p']} | "
          f"rank tau={rs['kendall_tau_mean']}")


if __name__ == "__main__":
    main()
