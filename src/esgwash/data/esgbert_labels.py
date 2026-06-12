"""Hoan thien nhan topic bang cross-inference ESGBERT (Schimanski et al. 2023, ACL Findings).

Buoc chuan bi data (spec 01 #3, vong 2): bang masked sau merge con o NaN
(cau cua tap nay khong co nhan cho tru khac). Dung 3 classifier da cong bo cua
ESGBERT (chinh la model train tren env/soc/gov_2k goc) infer tren text_en de dien
cac o do voi confidence >= tau -> bang nhan dich `topic_labeled.parquet`,
mac dinh cho train. Bang masked-only giu lai cho ablation.

Chong leak: chi du doan o (split=train, nhan=NaN) — model tru p chua tung thay
nhan p cua cau den tu tap khac; o da co nhan goc khong bi ghi de.
Nhan topic bat bien qua dich -> map sang VI qua row-alignment, khong can infer text VI.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from esgwash.models.trainer import get_device

ESGBERT_MODELS = {
    "env": "ESGBERT/EnvironmentalBERT-environmental",
    "soc": "ESGBERT/SocialBERT-social",
    "gov": "ESGBERT/GovernanceBERT-governance",
}
PILLARS = ("env", "soc", "gov")

MASKED_TABLE = Path("data/processed/gold/topic_masked.parquet")


@torch.no_grad()
def _pillar_probs(texts: list[str], model_name: str,
                  batch_size: int = 64, max_length: int = 256) -> np.ndarray:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = get_device()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device).eval()
    # positive = nhan khac 'none' trong id2label (vd {0: 'none', 1: 'environmental'})
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    pos_idx = next(i for i, lab in id2label.items() if lab != "none")

    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], truncation=True, padding=True,
                  max_length=max_length, return_tensors="pt").to(device)
        probs = torch.softmax(model(**enc).logits, dim=-1)[:, pos_idx]
        out.append(probs.cpu().numpy())
    del model
    return np.concatenate(out)


def esgbert_probs(df: pd.DataFrame, batch_size: int = 64) -> pd.DataFrame:
    """Xac suat positive per pillar tren text_en, chi tinh o (train, NaN).

    -> DataFrame cung index voi df, cot env/soc/gov; NaN o khong tinh
    (da co nhan goc hoac khong thuoc split train).
    """
    probs = pd.DataFrame(np.nan, index=df.index, columns=list(PILLARS))
    train = df["split"] == "train" if "split" in df else pd.Series(True, index=df.index)
    for p in PILLARS:
        need = df[p].isna() & train & df["text_en"].notna()
        if not need.any():
            continue
        texts = df.loc[need, "text_en"].astype(str).tolist()
        probs.loc[need, p] = _pillar_probs(texts, ESGBERT_MODELS[p], batch_size)
    return probs


def build_labeled_table(cfg: dict, batch_size: int = 64, tau: float | None = None,
                        force: bool = False) -> dict:
    """Stage cross_label: masked table -> probs (cache) -> dien nhan -> topic_labeled.parquet.

    cfg = config topic (can khoa cross_label). Tra ve dict thong ke.
    """
    from esgwash.data.topic_merge import fill_cross_labels, label_stats

    cl = cfg.get("cross_label", {})
    tau = tau if tau is not None else cl.get("confidence", 0.9)
    probs_path = Path(cl.get("probs_path", "data/processed/gold/esgbert_probs.parquet"))
    table_path = Path(cl.get("table_path", "data/processed/gold/topic_labeled.parquet"))

    df = pd.read_parquet(MASKED_TABLE)
    if probs_path.exists() and not force:
        probs = pd.read_parquet(probs_path)
        assert len(probs) == len(df), "cache probs khong align voi bang masked — chay force"
    else:
        probs = esgbert_probs(df, batch_size=batch_size)
        probs.to_parquet(probs_path, index=False)
    probs.index = df.index

    labeled = fill_cross_labels(df, probs, tau=tau)
    labeled.to_parquet(table_path, index=False)

    stats = {"tau": tau, "table": str(table_path)}
    for p in PILLARS:
        filled = labeled[p].notna() & df[p].isna()
        scored = probs[p].notna()
        stats[p] = {
            "n_scored": int(scored.sum()),
            "n_filled": int(filled.sum()),
            "fill_rate": round(float(filled.sum() / scored.sum()), 4) if scored.any() else 0.0,
            "n_pos": int((labeled.loc[filled, p] == 1).sum()),
            "n_neg": int((labeled.loc[filled, p] == 0).sum()),
        }
    stats["labels_before"] = label_stats(df)
    stats["labels_after"] = label_stats(labeled)

    out = Path("outputs/metrics/cross_label_stats.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    return stats
