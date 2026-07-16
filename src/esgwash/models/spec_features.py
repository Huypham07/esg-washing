"""Chunk-level structural features for the spec_level explainer (RQ5 / SHAP).

DELIBERATELY excludes bank/ticker/doc identifiers — encoding them leaks the
target (the reference SHAP notebook hit 100% accuracy via label leakage)."""
from __future__ import annotations

import re

import pandas as pd

PILLARS = ("env", "soc", "gov")
_ESG = [f"is_{p}" for p in PILLARS]
_DIGIT_RUN = re.compile(r"\d+")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")

FEATURE_COLS = ["token_count", "char_count", "word_count", "has_digit",
                "n_digit_runs", "has_year", "pct_digit_chars",
                "p_env", "p_soc", "p_gov", "p_commitment", "rel_position"]


def _esg_commit(classified: pd.DataFrame) -> pd.DataFrame:
    esg = classified[_ESG].max(axis=1).astype(bool)
    return classified[(classified["is_commitment"] == 1) & esg].reset_index(drop=True)


def chunk_features(classified: pd.DataFrame):
    df = _esg_commit(classified)
    text = df["content_text"].astype(str)
    digit_runs = text.apply(lambda s: _DIGIT_RUN.findall(s))
    n_digit_chars = text.apply(lambda s: sum(c.isdigit() for c in s))
    max_idx = df.groupby("doc_id")["chunk_index"].transform("max")
    rel = df["chunk_index"] / max_idx.where(max_idx > 0, 1)
    rel = rel.where(max_idx > 0, 0.0)

    X = pd.DataFrame({
        "token_count": df["token_count"].astype(float),
        "char_count": df["char_count"].astype(float),
        "word_count": text.str.split().apply(len).astype(float),
        "has_digit": digit_runs.apply(lambda r: int(len(r) > 0)),
        "n_digit_runs": digit_runs.apply(len).astype(float),
        "has_year": text.apply(lambda s: int(bool(_YEAR.search(s)))),
        "pct_digit_chars": (n_digit_chars / df["char_count"].clip(lower=1)).astype(float),
        "p_env": df["p_env"].astype(float),
        "p_soc": df["p_soc"].astype(float),
        "p_gov": df["p_gov"].astype(float),
        "p_commitment": df["p_commitment"].astype(float),
        "rel_position": rel.astype(float),
    }, columns=FEATURE_COLS)
    y = df["spec_level"].astype(int)
    return X, y
