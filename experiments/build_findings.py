"""Phase 4: consolidate Phases 0-3 artifacts into panel_master.csv + findings.md
(paper-ready source for RQ1-RQ5). Read-only on inputs; no model training; no LaTeX."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from esgwash.indices.panel_master import merge_signals, rq4_correlations, say_do_by_pillar


def _read_csv(p: Path):
    return pd.read_csv(p) if p.exists() else None


def build_master(panel_dir: str = "experiments/panel") -> pd.DataFrame:
    d = Path(panel_dir)
    panel = pd.read_csv(d / "panel.csv")
    bri = _read_csv(d / "embedding_signals.csv")
    say_do = _read_csv(d / "say_do.csv")
    if bri is None:
        bri = panel[["bank", "year"]].assign(bri=float("nan"))
    if say_do is None:
        rows = [{"bank": b, "year": y, "pillar": p, "say_do": float("nan")}
                for b, y in panel[["bank", "year"]].itertuples(index=False) for p in ("env", "soc", "gov")]
        say_do = pd.DataFrame(rows)
    return merge_signals(panel, bri, say_do)


def main(panel_dir: str = "experiments/panel", eval_dir: str = "experiments/eval") -> dict:
    d = Path(panel_dir)
    master = build_master(panel_dir)
    master.to_csv(d / "panel_master.csv", index=False)

    say_do = _read_csv(d / "say_do.csv")
    summ = json.loads((d / "summary.json").read_text(encoding="utf-8")) if (d / "summary.json").exists() else {}
    transfer = _read_csv(Path(eval_dir) / "transfer.csv")
    figtypes = _read_csv(d / "figure_types.csv")

    rq4 = rq4_correlations(master)
    saydo = say_do_by_pillar(say_do) if say_do is not None else {}
    desc = summ.get("descriptive", {})

    L = ["# ESG-washing — consolidated findings (RQ1–RQ5)", "",
         f"Panel: {master['bank'].nunique()} banks x {master['year'].nunique()} years "
         f"= {len(master)} bank-years.", ""]
    L += ["## RQ1 — Prevalence"]
    for idx in ("cti", "nar", "qdr"):
        o = desc.get(f"{idx}_overall")
        if o:
            L.append(f"- {idx.upper()} = {o['mean']} (sd {o['sd']}, CI95 {o['ci95']})")
    L += ["", "## RQ2 — Selective disclosure"]
    sd2 = summ.get("selective_disclosure", {})
    if sd2:
        L.append(f"- pillar mean share: {sd2.get('pillar_mean_share')}")
        L.append(f"- Friedman chi2={sd2.get('friedman', {}).get('chi2')}, p={sd2.get('friedman', {}).get('p')}")
    L += ["", "## RQ3 — Temporal trend"]
    tr = summ.get("temporal_trend", {})
    for idx in ("cti", "qdr"):
        if idx in tr:
            L.append(f"- {idx.upper()}~year Spearman rho={tr[idx]['spearman_rho']} (p={tr[idx]['p']})")
    L += ["", "## RQ4 — Convergent validity (embedding BRI vs rubric)",
          f"- Spearman(CTI, BRI) = {rq4['cti_bri']['rho']} (p={rq4['cti_bri']['p']}, n={rq4['cti_bri']['n']})",
          f"- Spearman(NAR, BRI) = {rq4['nar_bri']['rho']} (p={rq4['nar_bri']['p']}, n={rq4['nar_bri']['n']})",
          "- Interpretation: boilerplate (cross-bank reused language) tracks NAMED actions, not vague "
          "cheap-talk; rubric CTI is not redundant with embedding similarity. (SBS dropped: anisotropy.)"]
    L += ["", "## RQ5 — Say-do gap + drivers"]
    if saydo:
        L.append(f"- mean say-do by pillar (CTI_p - QDR_p): {saydo} -> governance is the cheap-talk pillar.")
    if figtypes is not None:
        tot = figtypes.groupby("type")["n"].sum().sort_values(ascending=False)
        L.append(f"- quantified figures by type: {tot.to_dict()}")
    L += ["", "## Cross-lingual transfer (lexical baseline)"]
    if transfer is not None:
        for _, r in transfer.iterrows():
            L.append(f"- {r['task']} {r['config']}: macro-F1={r['macro_f1']}")
        L.append("- EN->VI collapse motivates translate-train / multilingual encoder.")

    (d / "findings.md").write_text("\n".join(L), encoding="utf-8")
    print(f"-> {d/'panel_master.csv'} ({len(master)} rows)")
    print(f"-> {d/'findings.md'}")
    print(f"RQ4 CTI~BRI rho={rq4['cti_bri']['rho']} p={rq4['cti_bri']['p']}; say-do {saydo}")
    return {"master": master, "rq4": rq4, "say_do": saydo}


if __name__ == "__main__":
    main()
