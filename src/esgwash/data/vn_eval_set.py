"""Bộ VN human-eval (spec 01 #5) — test set chính để đánh giá transfer.

Lấy mẫu ~300 câu phân tầng -> CSV để gán nhãn; nạp lại nhãn + tính Cohen kappa.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANNOT_COLS = ["env", "soc", "gov", "commitment", "specificity"]


def sample_for_annotation(sentences_df: pd.DataFrame, n: int = 300, seed: int = 42,
                          preds: pd.DataFrame | None = None) -> pd.DataFrame:
    """Stratified theo bank (x tru du doan neu co preds: sent_id + cot env/soc/gov)."""
    df = sentences_df.copy()
    if preds is not None:
        pr = preds.set_index("sent_id")[["env", "soc", "gov"]]
        pillar = pr.idxmax(axis=1).where(pr.max(axis=1) >= 0.5, "non_esg")
        df["stratum"] = df["bank"] + "_" + df["sent_id"].map(pillar).fillna("non_esg")
    else:
        df["stratum"] = df["bank"]

    rng = np.random.default_rng(seed)
    groups = df.groupby("stratum")
    per_group = max(1, n // groups.ngroups)
    picked = []
    for _, g in groups:
        k = min(per_group, len(g))
        picked.append(g.iloc[rng.choice(len(g), size=k, replace=False)])
    out = pd.concat(picked, ignore_index=True)
    if len(out) > n:
        out = out.iloc[rng.choice(len(out), size=n, replace=False)].reset_index(drop=True)

    out = out[["sent_id", "doc_id", "bank", "year", "sentence", "ctx_prev", "ctx_next"]]
    for c in ANNOT_COLS:
        out[c] = ""
    return out.sort_values(["bank", "year", "sent_id"]).reset_index(drop=True)


def load_annotations(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ANNOT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=ANNOT_COLS, how="all")


def inter_annotator_kappa(ann_a: pd.DataFrame, ann_b: pd.DataFrame) -> dict:
    from sklearn.metrics import cohen_kappa_score
    merged = ann_a.merge(ann_b, on="sent_id", suffixes=("_a", "_b"))
    out = {}
    for c in ANNOT_COLS:
        sub = merged[[f"{c}_a", f"{c}_b"]].dropna()
        if len(sub):
            out[c] = {"kappa": round(float(
                cohen_kappa_score(sub[f"{c}_a"], sub[f"{c}_b"])), 4), "n": int(len(sub))}
    return out
