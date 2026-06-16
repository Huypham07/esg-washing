"""CTI / NAR / QDR tu thang specificity 3 muc (spec 2026-06-16).

Tren moi o (bank, year, pillar), denominator = cam ket co gan tru ESG do:
  CTI = ti le spec_level 0 (mo ho / cheap talk)   -> truc washing
  NAR = ti le spec_level 1 (hanh dong co ten)      -> vung xam
  QDR = ti le spec_level 2 (dinh luong, quy ve chu the) -> substance
CTI+NAR+QDR = 1. Chi la ti le output classifier, khong tu dat trong so.
Bo hoan toan grounded-CTI (xem legacy/README.md).

Input `claims_long`: moi dong = (cam ket x tru no thuoc ve), cot
bank, year, pillar, is_commitment, spec_level.
"""
from __future__ import annotations

import pandas as pd

from esgwash.indices.bootstrap import bootstrap_ci

CELL = ["bank", "year", "pillar"]
_LEVEL_COL = {"cti": 0, "nar": 1, "qdr": 2}


def _share_of_level(level: int):
    def stat(v):
        return float((v == level).mean())
    return stat


def compute_specificity_shares(claims_long: pd.DataFrame, n_resamples: int = 1000,
                               ci: float = 0.95, seed: int = 42) -> pd.DataFrame:
    """Bang share 3 muc + bootstrap CI cho moi o (bank, year, pillar)."""
    commit = claims_long[claims_long["is_commitment"] == 1]
    rows = []
    for (b, y, p), g in commit.groupby(CELL):
        lv = g["spec_level"].to_numpy(dtype=float)
        rec = {"bank": b, "year": y, "pillar": p, "n_commit": len(g)}
        for name, level in _LEVEL_COL.items():
            point, lo, hi = bootstrap_ci(lv, _share_of_level(level), n_resamples, ci, seed)
            rec[name] = round(point, 4)
            rec[f"{name}_lo"] = round(lo, 4)
            rec[f"{name}_hi"] = round(hi, 4)
        rows.append(rec)
    return pd.DataFrame(rows)


def build_index_table(claims_long: pd.DataFrame, n_resamples: int = 1000,
                      ci: float = 0.95, seed: int = 42) -> pd.DataFrame:
    """Alias on dinh ten cho orchestrator; co the merge them cot mo ta sau."""
    return compute_specificity_shares(claims_long, n_resamples, ci, seed)
