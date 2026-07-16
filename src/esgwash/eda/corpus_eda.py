"""Pure compute for corpus-level EDA (no plotting)."""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

from esgwash.eda.style import effective_states, shannon_entropy_bits

PILLARS = ("env", "soc", "gov")
_ESG = [f"is_{p}" for p in PILLARS]


def load_classified(root: str = "outputs/cti") -> pd.DataFrame:
    files = sorted(glob.glob(str(Path(root) / "*/*/classified.parquet")))
    frames = [pd.read_parquet(f) for f in files]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def token_distribution(chunks: pd.DataFrame, col: str = "token_count") -> dict:
    s = chunks[col].astype(float)
    q = s.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
    return {"p10": float(q.loc[0.10]), "q1": float(q.loc[0.25]),
            "median": float(q.loc[0.50]), "q3": float(q.loc[0.75]),
            "p90": float(q.loc[0.90]), "max": float(s.max()), "mean": float(s.mean())}


def label_positive_rates(clf: pd.DataFrame) -> pd.DataFrame:
    cols = _ESG + ["is_commitment"]
    return clf.groupby(["bank", "year"])[cols].mean().reset_index()


def _esg_commitment(clf: pd.DataFrame) -> pd.DataFrame:
    esg = clf[_ESG].max(axis=1).astype(bool)
    return clf[(clf["is_commitment"] == 1) & esg]


def spec_level_distribution(clf: pd.DataFrame) -> dict:
    sub = _esg_commitment(clf)
    counts = {int(k): int(v) for k, v in sub["spec_level"].value_counts().sort_index().items()}
    vals = list(counts.values())
    return {"counts": counts, "entropy_bits": shannon_entropy_bits(vals),
            "effective_states": effective_states(vals)}


def coverage_matrix(clf: pd.DataFrame, value: str = "chunk") -> pd.DataFrame:
    if value == "commitment":
        g = clf[clf["is_commitment"] == 1]
    else:
        g = clf
    # Use any column that exists for counting (chunk_id if available, else any other column)
    count_col = "chunk_id" if "chunk_id" in g.columns else g.columns[0]
    return g.pivot_table(index="bank", columns="year", values=count_col,
                         aggfunc="count", fill_value=0)
