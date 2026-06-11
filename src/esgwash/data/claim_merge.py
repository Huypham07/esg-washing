"""Gop du lieu cho M2 (spec 02 #2): bang multi-head voi nhan partial.

Heads: commitment, specificity (cung tap van ban - merge tu gold_loader);
aux head claim (env_claims, chi regularize encoder);
augment commitment: action_500 (ESG-wide) + ml_promise (khi co ban dich).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from esgwash.data.gold_loader import (load_action, load_claim_pair, load_env_claims,
                                      load_ml_promise)

HEADS = ("commitment", "specificity", "claim")


def _carve_dev(df: pd.DataFrame, frac: float = 0.1, seed: int = 42) -> pd.DataFrame:
    train = df[df["split"] == "train"]
    strat = (train["commitment"].fillna(-1).astype(int).astype(str)
             + train["specificity"].fillna(-1).astype(int).astype(str))
    _, dev_idx = train_test_split(train.index, test_size=frac, stratify=strat,
                                  random_state=seed)
    df = df.copy()
    df.loc[dev_idx, "split"] = "dev"
    return df


def build_claim_table(lang: str = "vi", augment_action: bool = True,
                      aux_env_claims: bool = True, augment_ml_promise: bool = False,
                      seed: int = 42) -> pd.DataFrame:
    """-> DataFrame[text, text_en, commitment, specificity, claim, source, split]."""
    base = load_claim_pair(lang)
    base["claim"] = np.nan
    base["source"] = "climatebert"
    base = _carve_dev(base, seed=seed)
    parts = [base]

    if augment_action:
        act = load_action(lang)
        parts.append(pd.DataFrame({
            "text": act["text"], "text_en": act["text_en"],
            "commitment": act["action"].astype(float), "specificity": np.nan,
            "claim": np.nan, "source": "action_500", "split": "train"}))

    if aux_env_claims:
        ec = load_env_claims(lang)
        ec = ec[ec["split"] == "train"]
        parts.append(pd.DataFrame({
            "text": ec["text"], "text_en": ec["text_en"],
            "commitment": np.nan, "specificity": np.nan,
            "claim": ec["claim"].astype(float), "source": "env_claims",
            "split": "train"}))

    if augment_ml_promise:
        mp = load_ml_promise(lang)
        parts.append(pd.DataFrame({
            "text": mp["text"], "text_en": np.nan,
            "commitment": mp["promise"].astype(float), "specificity": np.nan,
            "claim": np.nan, "source": "ml_promise_" + mp["lang_src"],
            "split": "train"}))

    out = pd.concat(parts, ignore_index=True)
    return out[["text", "text_en", "commitment", "specificity", "claim", "source", "split"]]


def label_stats(df: pd.DataFrame) -> dict:
    out = {"n_rows": int(len(df)), "sources": df["source"].value_counts().to_dict()}
    for h in HEADS:
        sub = df[h].dropna()
        out[h] = {"n_labeled": int(len(sub)), "pos_rate": round(float(sub.mean()), 4)}
    out["split_sizes"] = df["split"].value_counts().to_dict()
    return out
