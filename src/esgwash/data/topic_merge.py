"""Gộp 3 tập topic nhị phân thành bảng masked multi-label (spec 01 #3).

Các tập chia sẻ ~50-60% văn bản (gộp theo text_en để không nhân đôi);
cột thiếu nhãn = NaN -> masked BCE. Không bịa nhãn.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from esgwash.data.gold_loader import load_env_claims, load_topic

PILLARS = ("env", "soc", "gov")


def _resolve_dups(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Cùng text_en xuất hiện nhiều lần: nhãn mâu thuẫn thì bỏ, còn lại giữ dòng đầu."""
    conflict = df.groupby("text_en")[label].transform("nunique") > 1
    return df[~conflict].drop_duplicates("text_en")


def build_masked_table(lang: str = "vi") -> pd.DataFrame:
    """-> [text, text_en, env, soc, gov, sources]; mỗi dòng là một text duy nhất."""
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
    """Chia split trên text duy nhất (không leak), stratify theo pattern nhãn khả dụng."""
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
    """Cắt val từ train ngay lúc train (data trên đĩa không còn cột split).

    Stratify theo pattern nhãn khả dụng; `mask` giới hạn pool lấy val (vd commitment
    chỉ lấy climatebert để val cùng phân phối với test). Train = phần còn lại (gồm cả
    các dòng ngoài mask), val = phần cắt ra. Không leak vì mỗi dòng là một text duy nhất.
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
    """Positive của env_claims chắc chắn là E-positive -> thêm env=1
    (chỉ train; soc/gov để NaN cho fill_cross_labels điền tiếp)."""
    ec = load_env_claims(lang)
    pos = ec[(ec["claim"] == 1) & (ec["split"] == "train")]
    pos = pos[~pos["text_en"].isin(df["text_en"])]
    add = pd.DataFrame({"text": pos["text"], "text_en": pos["text_en"],
                        "env": 1.0, "soc": np.nan, "gov": np.nan,
                        "sources": "env_claims", "split": "train"})
    return pd.concat([df, add], ignore_index=True)


def build_topic_table(lang: str = "vi", seed: int = 42) -> pd.DataFrame:
    """Bảng masked: gộp 3 tập topic -> chia split -> nhập env_claims positives.
    Ô NaN được điền bằng ESGBERT cross-inference ở bước riêng (fill_cross_labels)."""
    df = build_masked_table(lang)
    df = split_stratified(df, seed=seed)
    return _append_env_claims(df, lang=lang)


def fill_cross_labels(
    df: pd.DataFrame,
    probs: pd.DataFrame,
    tau: float = 0.9,
) -> pd.DataFrame:
    """Điền ô NaN bằng dự đoán ESGBERT có confidence >= tau (cả positive lẫn negative).

    Chỉ điền ở pool train (split != 'test'); test giữ NaN ở trụ không có gold gốc nên
    test luôn đo trên gold thuần. val cắt từ train (carve_val) nên cũng được điền (silver,
    chấp nhận vì val chỉ để tune threshold/early-stop). Không có cột split -> điền hết.
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
