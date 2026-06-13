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
    idx_val, idx_test = train_test_split(
        idx_rest, test_size=ratios[2] / (ratios[1] + ratios[2]),
        stratify=pattern[idx_rest], random_state=seed)
    df = df.copy()
    df["split"] = "train"
    df.loc[idx_val, "split"] = "val"
    df.loc[idx_test, "split"] = "test"
    return df


def carve_val(df: pd.DataFrame, label_cols, seed: int = 42,
              val_frac: float = 0.1, mask=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cat val tu train ngay luc train (data tren dia khong con cot split).

    Stratify theo pattern nhan kha dung; `mask` gioi han pool lay val (vd claim:
    chi climatebert de val cung phan phoi voi test). Train = phan con lai (gom ca
    cac dong ngoai mask), val = phan cat ra. Khong leak vi moi dong la 1 text duy nhat.
    """
    pool = df.index if mask is None else df.index[mask]
    pattern = df.loc[pool, list(label_cols)].fillna(-1).astype(int).astype(str).agg("".join, axis=1)
    rare = pattern.value_counts()
    pattern = pattern.where(pattern.map(rare) >= 10, "rare")
    _, val_idx = train_test_split(pool, test_size=val_frac, stratify=pattern,
                                  random_state=seed)
    val = df.loc[val_idx]
    train = df.drop(index=val_idx)
    return train, val


def _append_env_claims(df: pd.DataFrame, lang: str = "vi") -> pd.DataFrame:
    """Positive cua env_claims la E-positive chac chan -> them env=1
    (train-only; soc/gov NaN de cross_label dien tiep)."""
    ec = load_env_claims(lang)
    pos = ec[(ec["claim"] == 1) & (ec["split"] == "train")]
    pos = pos[~pos["text_en"].isin(df["text_en"])]
    add = pd.DataFrame({"text": pos["text"], "text_en": pos["text_en"],
                        "env": 1.0, "soc": np.nan, "gov": np.nan,
                        "sources": "env_claims", "split": "train"})
    return pd.concat([df, add], ignore_index=True)


def build_topic_table(lang: str = "vi", seed: int = 42) -> pd.DataFrame:
    """Bang masked hoan chinh cho stage prepare_gold:
    merge 3 tap -> split -> nhap env_claims positives. (Buoc dien NaN bang ESGBERT
    da archive vao unused/esgbert_labels.py — topic build da dong bang 2026-06-13.)"""
    df = build_masked_table(lang)
    df = split_stratified(df, seed=seed)
    return _append_env_claims(df, lang=lang)


def fill_cross_labels(
    df: pd.DataFrame,
    probs: pd.DataFrame,
    tau: float = 0.9,
) -> pd.DataFrame:
    """Dien o NaN bang du doan ESGBERT confidence >= tau (positive lan negative).

    CHI dien o pool train (split != 'test'); test giu NaN o tru khong co gold goc
    -> evaluate() tu mask NaN nen test do tren GOLD thuan (quyet dinh user 2026-06-13).
    val cat tu train luc train (carve_val) nen cung nam trong pool duoc dien (silver,
    chap nhan duoc vi val chi de tune threshold/early-stop). Khong co cot split -> dien het.
    """
    df = df.copy()
    fillable = df["split"] != "test" if "split" in df.columns else pd.Series(True, index=df.index)

    for p in PILLARS:
        na = df[p].isna() & fillable

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
