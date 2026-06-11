"""Merge 3 tap topic nhi phan -> bang masked multi-label (spec 01 #3).

Cac tap chia se ~50-60% van ban (merge theo text_en de khong nhan doi);
cot thieu nhan = NaN -> masked BCE. Khong bia nhan.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from esgwash.data.gold_loader import load_env_claims, load_topic

PILLARS = ("env", "soc", "gov")


def _resolve_dups(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Cung text_en xuat hien nhieu lan: nhan mau thuan -> bo; con lai giu dong dau."""
    conflict = df.groupby("text_en")[label].transform("nunique") > 1
    return df[~conflict].drop_duplicates("text_en")


def build_masked_table(lang: str = "vi") -> pd.DataFrame:
    """-> DataFrame[text, text_en, env, soc, gov, sources] - moi dong la 1 text duy nhat."""
    merged = None
    vi_texts: dict[str, str] = {}
    for p in PILLARS:
        df = _resolve_dups(load_topic(p, lang), p)
        for ten, tvi in zip(df["text_en"], df["text"]):
            vi_texts.setdefault(ten, tvi)
        part = df[["text_en", p]]
        merged = part if merged is None else merged.merge(part, on="text_en", how="outer")
    merged["text"] = merged["text_en"].map(vi_texts)
    merged["sources"] = merged[list(PILLARS)].notna().dot(
        pd.Index(PILLARS) + "+").str.rstrip("+")
    return merged[["text", "text_en", *PILLARS, "sources"]]


def split_stratified(df: pd.DataFrame, seed: int = 42,
                     ratios: tuple = (0.8, 0.1, 0.1)) -> pd.DataFrame:
    """Split tren text duy nhat (khong leak), stratify theo pattern nhan kha dung."""
    pattern = df[list(PILLARS)].fillna(-1).astype(int).astype(str).agg("".join, axis=1)
    rare = pattern.value_counts()
    pattern = pattern.where(pattern.map(rare) >= 10, "rare")
    idx_train, idx_rest = train_test_split(
        df.index, test_size=ratios[1] + ratios[2], stratify=pattern, random_state=seed)
    idx_dev, idx_test = train_test_split(
        idx_rest, test_size=ratios[2] / (ratios[1] + ratios[2]),
        stratify=pattern[idx_rest], random_state=seed)
    df = df.copy()
    df["split"] = "train"
    df.loc[idx_dev, "split"] = "dev"
    df.loc[idx_test, "split"] = "test"
    return df


def augment_env_claims(df: pd.DataFrame, lang: str = "vi") -> pd.DataFrame:
    """Positive cua env_claims la E-positive chac chan -> them env=1 (train only, ablation)."""
    ec = load_env_claims(lang)
    pos = ec[(ec["claim"] == 1) & (ec["split"] == "train")]
    pos = pos[~pos["text_en"].isin(df["text_en"])]
    add = pd.DataFrame({"text": pos["text"], "text_en": pos["text_en"],
                        "env": 1.0, "soc": np.nan, "gov": np.nan,
                        "sources": "env_claims", "split": "train"})
    return pd.concat([df, add], ignore_index=True)


def cross_pseudo_label(df: pd.DataFrame, probs: pd.DataFrame, tau: float = 0.9) -> pd.DataFrame:
    """Vong 2 optional: dien NaN khi model trai nguon du doan confidence >= tau.

    probs: DataFrame cung index voi df, cot env/soc/gov = xac suat du doan.
    Chi dien cho split=train; nhan goc khong bi ghi de.
    """
    df = df.copy()
    train = df["split"] == "train"
    for p in PILLARS:
        na = df[p].isna() & train
        conf_pos = na & (probs[p] >= tau)
        conf_neg = na & (probs[p] <= 1 - tau)
        df.loc[conf_pos, p] = 1.0
        df.loc[conf_neg, p] = 0.0
    return df


def label_stats(df: pd.DataFrame) -> dict:
    out = {"n_rows": int(len(df))}
    for p in PILLARS:
        sub = df[p].dropna()
        out[p] = {"n_labeled": int(len(sub)), "pos_rate": round(float(sub.mean()), 4)}
    if "split" in df:
        out["split_sizes"] = df["split"].value_counts().to_dict()
    return out
