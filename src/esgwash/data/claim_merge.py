"""Gop du lieu cho M2 (spec 02 #2): bang multi-head voi nhan partial.

Heads: commitment, specificity (cung tap van ban - merge tu gold_loader);
augment commitment: action_500 (ESG-wide) + ml_promise (EN+FR+JA dich VI).
Aux head env_claims da BO (quyet dinh 2026-06-12) — env_claims chi con dung
cho topic-E (nhap mac dinh trong topic_merge.build_topic_table).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from esgwash.data.gold_loader import load_action, load_claim_pair, load_ml_promise

HEADS = ("commitment", "specificity")


def _carve_val(df: pd.DataFrame, frac: float = 0.1, seed: int = 42) -> pd.DataFrame:
    train = df[df["split"] == "train"]
    strat = (train["commitment"].fillna(-1).astype(int).astype(str)
             + train["specificity"].fillna(-1).astype(int).astype(str))
    _, val_idx = train_test_split(train.index, test_size=frac, stratify=strat,
                                  random_state=seed)
    df = df.copy()
    df.loc[val_idx, "split"] = "val"
    return df


def build_claim_table(lang: str = "vi", augment_action: bool = True,
                      augment_ml_promise: bool = True, seed: int = 42) -> pd.DataFrame:
    """-> DataFrame[text, text_en, commitment, specificity, source, split]."""
    base = load_claim_pair(lang)
    base["source"] = "climatebert"
    base = _carve_val(base, seed=seed)
    parts = [base]

    if augment_action:
        act = load_action(lang)
        parts.append(pd.DataFrame({
            "text": act["text"], "text_en": act["text_en"],
            "commitment": act["action"].astype(float), "specificity": np.nan,
            "source": "action_500", "split": "train"}))

    if augment_ml_promise:
        mp = load_ml_promise(lang)
        parts.append(pd.DataFrame({
            "text": mp["text"], "text_en": np.nan,
            "commitment": mp["promise"].astype(float), "specificity": np.nan,
            "source": "ml_promise_" + mp["lang_src"], "split": "train"}))

    out = pd.concat(parts, ignore_index=True)
    return out[["text", "text_en", "commitment", "specificity", "source", "split"]]


def label_stats(df: pd.DataFrame) -> dict:
    out = {"n_rows": int(len(df)), "sources": df["source"].value_counts().to_dict()}
    for h in HEADS:
        sub = df[h].dropna()
        out[h] = {"n_labeled": int(len(sub)), "pos_rate": round(float(sub.mean()), 4)}
    out["split_sizes"] = df["split"].value_counts().to_dict()
    return out
