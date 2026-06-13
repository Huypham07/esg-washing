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
    probs = pd.DataFrame(np.nan, index=df.index, columns=list(PILLARS))

    for p in PILLARS:
        need = df[p].isna() & df["text_en"].notna()

        if not need.any():
            continue

        texts = df.loc[need, "text_en"].astype(str).tolist()
        probs.loc[need, p] = _pillar_probs(
            texts,
            ESGBERT_MODELS[p],
            batch_size,
        )

    return probs


def build_labeled_table(cfg: dict, batch_size: int = 64, tau: float | None = None) -> dict:
    from esgwash.data.topic_merge import fill_cross_labels, label_stats

    cl = cfg.get("cross_label", {})
    tau = tau if tau is not None else cl.get("confidence", 0.9)
    probs_path = Path(cl.get("probs_path", "data/processed/gold/esgbert_probs.parquet"))
    table_path = Path(cl.get("table_path", "data/processed/gold/topic_labeled.parquet"))

    df = pd.read_parquet(MASKED_TABLE)

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
