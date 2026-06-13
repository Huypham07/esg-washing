"""Chạy 5 binary classifier (silver|gold) trên corpus -> enriched parquet cho CTI.

Thứ tự gate (giống pipeline thiết kế):
  3 TOPIC binary (env/soc/gov) trên TẤT CẢ câu  -> is_E/is_S/is_G (chấm chéo, 1 câu vào >=0 pillar)
  COMMITMENT chỉ trên câu ESG (is_esg=1)        -> commitment
  SPECIFICITY chỉ trên câu commit=1             -> specificity

Chạy:
  python src/pipeline/classify_corpus.py                      # full corpus, track theo config
  python src/pipeline/classify_corpus.py --limit 300 --device cpu   # smoke
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, logging as hf_logging

sys.stdout.reconfigure(encoding="utf-8")  # tránh crash print tiếng Việt trên cp1252
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.corpus.word_segment import word_segment_batch

hf_logging.set_verbosity_error()


def load_cti_config(path: str = "config/cti.yml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_device(device: str | None) -> torch.device:
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _classify_texts(model_path: str, texts: list[str], max_length: int,
                    batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """-> (preds 0/1, prob_lớp1). Nạp model -> infer batched -> giải phóng."""
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval()

    n = len(texts)
    preds = np.zeros(n, dtype=np.int64)
    prob1 = np.zeros(n, dtype=np.float32)
    tag = Path(model_path).parent.name  # vd topic_e_silver
    for i in tqdm(range(0, n, batch_size), desc=f"  {tag}", leave=False):
        batch = texts[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        sm = torch.softmax(logits, dim=-1)
        preds[i:i + len(batch)] = sm.argmax(dim=-1).cpu().numpy()
        prob1[i:i + len(batch)] = sm[:, 1].cpu().numpy()  # lớp index 1 = label positive

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return preds, prob1


def classify_corpus(config: dict, track: str | None = None, limit: int = 0,
                    device: str | None = None, use_cache: bool = True) -> pd.DataFrame:
    track = track or config["track"]
    models = config["models"][track]
    bs = int(config["batch_size"])
    ml_topic = int(config["max_length"]["topic"])
    ml_subst = int(config["max_length"]["subst"])
    dev = _resolve_device(device)

    out_dir = Path(config["paths"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (f"_smoke_enriched_{track}.parquet" if limit else f"enriched_corpus_{track}.parquet")

    if use_cache and out_path.exists() and not limit:
        print(f"[cache] {out_path} đã có -> đọc lại (--no-cache để ép chạy lại)")
        return pd.read_parquet(out_path)

    df = pd.read_parquet(config["paths"]["corpus"])
    if limit:
        df = df.head(limit)
    df = df.reset_index(drop=True)
    print(f"Corpus: {len(df):,} câu | track={track} | device={dev}")

    # Segment 1 LẦN -> tái dùng cho cả 5 model (underthesea chậm, đừng segment lặp)
    if config.get("word_segment", True):
        print("Word-segment (underthesea)...")
        seg = word_segment_batch(df["sentence"].astype(str).tolist())
    else:
        seg = df["sentence"].astype(str).tolist()

    # --- TOPIC: 3 binary trên TẤT CẢ câu (chấm chéo) ---
    print("Topic (env/soc/gov)...")
    for pillar, key in [("E", "env"), ("S", "soc"), ("G", "gov")]:
        preds, p1 = _classify_texts(models[key], seg, ml_topic, bs, dev)
        df[f"is_{pillar}"] = preds
        df[f"p_{pillar}"] = p1
    df["is_esg"] = (df[["is_E", "is_S", "is_G"]].max(axis=1) == 1).astype(int)
    print(f"  -> E={int(df.is_E.sum())} S={int(df.is_S.sum())} G={int(df.is_G.sum())} "
          f"| ESG={int(df.is_esg.sum())}/{len(df)}")

    # --- COMMITMENT chỉ trên câu ESG ---
    print("Commitment (câu ESG)...")
    df["commitment"] = np.nan
    df["p_commit"] = np.nan
    esg_idx = df.index[df.is_esg == 1].tolist()
    if esg_idx:
        preds, p1 = _classify_texts(models["commitment"], [seg[i] for i in esg_idx], ml_subst, bs, dev)
        df.loc[esg_idx, "commitment"] = preds
        df.loc[esg_idx, "p_commit"] = p1
    print(f"  -> commit=1: {int((df.commitment == 1).sum())} / ESG {len(esg_idx)}")

    # --- SPECIFICITY chỉ trên câu commit=1 ---
    print("Specificity (câu commit=1)...")
    df["specificity"] = np.nan
    df["p_spec"] = np.nan
    com_idx = df.index[df.commitment == 1].tolist()
    if com_idx:
        preds, p1 = _classify_texts(models["specificity"], [seg[i] for i in com_idx], ml_subst, bs, dev)
        df.loc[com_idx, "specificity"] = preds
        df.loc[com_idx, "p_spec"] = p1
    print(f"  -> spec=1: {int((df.specificity == 1).sum())} / commit {len(com_idx)}")

    keep = ["doc_id", "bank", "year", "sent_id", "section_title", "block_type", "sentence",
            "is_E", "is_S", "is_G", "p_E", "p_S", "p_G", "is_esg",
            "commitment", "p_commit", "specificity", "p_spec"]
    out = df[[c for c in keep if c in df.columns]]
    out.to_parquet(out_path, index=False)
    print(f"Saved enriched -> {out_path}  ({len(out):,} câu)")
    return out


def parse_args(args=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Classify corpus với 5 binary model -> enriched parquet")
    p.add_argument("--config", default="config/cti.yml")
    p.add_argument("--track", default=None, help="silver | gold (mặc định theo config)")
    p.add_argument("--limit", type=int, default=0, help="0 = full corpus; >0 = smoke N câu đầu")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda")
    p.add_argument("--no-cache", action="store_true")
    return p.parse_args(args)


def main(args=None) -> None:
    a = parse_args(args)
    config = load_cti_config(a.config)
    classify_corpus(config, track=a.track, limit=a.limit, device=a.device, use_cache=not a.no_cache)


if __name__ == "__main__":
    main()
