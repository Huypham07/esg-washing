"""CTI / NAR / QDR tu thang specificity 3 muc (spec 2026-06-16).

Tren moi o (bank, year), denominator = tap hop UNIQUE cam ket co it nhat 1 tru ESG duong:
  CTI = ti le spec_level 0 (mo ho / cheap talk)          -> truc washing
  NAR = ti le spec_level 1 (hanh dong co ten, chua dinh luong)
  QDR = ti le spec_level 2 (dinh luong, quy ve chu the)  -> substance
CTI+NAR+QDR = 1. Khong tach theo tru de tranh double-count; phan tich pillar luu rieng
o pillar_shares.parquet. Bo hoan toan grounded-CTI (xem legacy/README.md).

Input `classified`: DataFrame toan bo chunk da classify, cot
bank, year, is_env, is_soc, is_gov, is_commitment, spec_level.
"""
from __future__ import annotations

import pandas as pd

from esgwash.indices.bootstrap import bootstrap_ci

CELL = ["bank", "year"]
_LEVEL_COL = {"cti": 0, "nar": 1, "qdr": 2}
_ESG_PILLARS = ("is_env", "is_soc", "is_gov")

INDEX_LEGEND: dict[str, str] = {
    "cti":        "Cheap Talk Index — share of ESG commitment chunks rated vague (spec_level=0); higher = more washing.",
    "nar":        "Named Action Rate — share citing a named action/tool (spec_level=1) but unquantified.",
    "qdr":        "Quantified Disclosure Rate — share with a figure attributable to the bank (spec_level=2).",
    "n_commit":   "Unique ESG commitment chunks in (bank, year); denominator for CTI/NAR/QDR.",
    "spec_level": "0=vague, 1=named action, 2=quantified. CTI+NAR+QDR=1.",
}


def _share_of_level(level: int):
    def stat(v):
        return float((v == level).mean())
    return stat


def compute_specificity_shares(classified: pd.DataFrame, n_resamples: int = 1000,
                               ci: float = 0.95, seed: int = 42) -> pd.DataFrame:
    """CTI/NAR/QDR per (bank, year) tren unique ESG commitment chunks (khong double-count theo tru)."""
    esg_mask = classified[list(_ESG_PILLARS)].max(axis=1).astype(bool)
    commit = classified[(classified["is_commitment"] == 1) & esg_mask]
    rows = []
    for (b, y), g in commit.groupby(CELL):
        lv = g["spec_level"].to_numpy(dtype=float)
        rec = {"bank": b, "year": y, "n_commit": len(g)}
        for name, level in _LEVEL_COL.items():
            point, lo, hi = bootstrap_ci(lv, _share_of_level(level), n_resamples, ci, seed)
            rec[name] = round(point, 4)
            rec[f"{name}_lo"] = round(lo, 4)
            rec[f"{name}_hi"] = round(hi, 4)
        rows.append(rec)
    cols = ["bank", "year", "n_commit"] + [c for name in _LEVEL_COL for c in (name, f"{name}_lo", f"{name}_hi")]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def build_index_table(classified: pd.DataFrame, n_resamples: int = 1000,
                      ci: float = 0.95, seed: int = 42) -> pd.DataFrame:
    return compute_specificity_shares(classified, n_resamples, ci, seed)
