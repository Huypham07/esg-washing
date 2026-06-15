"""Gộp bảng huấn luyện cho đầu commitment (single-head).

Nguồn chính là commitments_actions (ClimateBERT); thêm action_500 và ml_promise
để phủ trụ S/G. Cột: text, text_en, commitment, source, split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from esgwash.data.gold_loader import load_action, load_commitment, load_ml_promise


def _carve_val(df: pd.DataFrame, frac: float = 0.1, seed: int = 42) -> pd.DataFrame:
    train = df[df["split"] == "train"]
    strat = train["commitment"].fillna(-1).astype(int)
    _, val_idx = train_test_split(train.index, test_size=frac, stratify=strat, random_state=seed)
    df = df.copy()
    df.loc[val_idx, "split"] = "val"
    return df


def build_commitment_table(lang: str = "vi", augment_action: bool = True,
                           augment_ml_promise: bool = True, seed: int = 42) -> pd.DataFrame:
    base = load_commitment(lang)
    base["source"] = "climatebert"
    base = _carve_val(base, seed=seed)
    parts = [base]

    if augment_action:
        act = load_action(lang)
        parts.append(pd.DataFrame({
            "text": act["text"], "text_en": act["text_en"],
            "commitment": act["action"].astype(float), "source": "action_500", "split": "train"}))

    if augment_ml_promise:
        mp = load_ml_promise(lang)
        parts.append(pd.DataFrame({
            "text": mp["text"], "text_en": np.nan,
            "commitment": mp["promise"].astype(float),
            "source": "ml_promise_" + mp["lang_src"], "split": "train"}))

    out = pd.concat(parts, ignore_index=True)
    return out[["text", "text_en", "commitment", "source", "split"]]


def label_stats(df: pd.DataFrame) -> dict:
    sub = df["commitment"].dropna()
    out = {"n_rows": int(len(df)), "sources": df["source"].value_counts().to_dict(),
           "commitment": {"n_labeled": int(len(sub)), "pos_rate": round(float(sub.mean()), 4)}}
    if "split" in df:
        out["split_sizes"] = df["split"].value_counts().to_dict()
    return out
