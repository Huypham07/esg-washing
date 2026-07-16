"""Merge all per-(bank,year) washing signals into one master panel + headline stats.

Signals: rubric CTI/NAR/QDR (panel.csv), embedding BRI (embedding_signals.csv),
per-pillar say-do gap (say_do.csv). Used to write the RQ1-RQ5 findings report.
"""
from __future__ import annotations

import pandas as pd
from scipy.stats import spearmanr

PILLARS = ("env", "soc", "gov")


def merge_signals(panel: pd.DataFrame, bri: pd.DataFrame, say_do: pd.DataFrame) -> pd.DataFrame:
    m = panel.merge(bri[["bank", "year", "bri"]], on=["bank", "year"], how="left")
    wide = say_do.pivot_table(index=["bank", "year"], columns="pillar",
                              values="say_do").reset_index()
    wide = wide.rename(columns={p: f"say_do_{p}" for p in PILLARS})
    for p in PILLARS:
        if f"say_do_{p}" not in wide.columns:
            wide[f"say_do_{p}"] = float("nan")
    keep = ["bank", "year"] + [f"say_do_{p}" for p in PILLARS]
    return m.merge(wide[keep], on=["bank", "year"], how="left")


def _spear(df: pd.DataFrame, a: str, b: str) -> dict:
    v = df.dropna(subset=[a, b])
    if len(v) < 3:
        return {"rho": float("nan"), "p": float("nan"), "n": int(len(v))}
    rho, p = spearmanr(v[a], v[b])
    return {"rho": round(float(rho), 4), "p": float(f"{p:.3e}"), "n": int(len(v))}


def rq4_correlations(master: pd.DataFrame) -> dict:
    return {"cti_bri": _spear(master, "cti", "bri"),
            "nar_bri": _spear(master, "nar", "bri")}


def say_do_by_pillar(say_do: pd.DataFrame) -> dict:
    g = say_do.groupby("pillar")["say_do"].mean()
    return {p: round(float(g.get(p, float("nan"))), 4) for p in PILLARS}
