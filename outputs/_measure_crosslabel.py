"""Đo lợi ích cross-label trước khi quyết (Phase 01): ESGBERT gán bao nhiêu POSITIVE mới?

Câu hỏi: với 5 model nhị phân rời, cross-label thêm được bao nhiêu nhãn-1 THẬT vào train
-> đáng làm không, hay chủ yếu thêm negative (vô ích cho recall task yếu)?

Chỉ ĐO trên các ô NaN (câu ngoài tập của trụ đó); KHÔNG ghi data train.
Chạy: python outputs/_measure_crosslabel.py
"""
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Console Windows = cp1252 không mã hoá được tiếng Việt -> ép UTF-8 cho mọi print.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TOPIC_DIR = Path("data/en_gold/topic")
FILES = {"env": "environmental_2k.csv", "soc": "social_2k.csv", "gov": "governance_2k.csv"}
ESGBERT = {
    "env": "ESGBERT/EnvironmentalBERT-environmental",
    "soc": "ESGBERT/SocialBERT-social",
    "gov": "ESGBERT/GovernanceBERT-governance",
}
PILLARS = ("env", "soc", "gov")
TAU = 0.9  # ngưỡng confidence như plan


def nfc(s) -> str:
    return unicodedata.normalize("NFC", str(s)).strip()


def build_masked() -> pd.DataFrame:
    """Merge 3 tập topic theo text (EN) -> bảng masked env/soc/gov (NaN = chưa nhãn)."""
    merged = None
    for p in PILLARS:
        df = pd.read_csv(TOPIC_DIR / FILES[p])
        df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
        df["text"] = df["text"].map(nfc)
        df = df.drop_duplicates("text")[["text", p]]
        merged = df if merged is None else merged.merge(df, on="text", how="outer")
    return merged


@torch.no_grad()
def esgbert_pos_prob(texts: list[str], model_name: str, bs: int = 64) -> np.ndarray:
    """P(positive) per câu — positive = nhãn khác 'none' trong id2label."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    m = AutoModelForSequenceClassification.from_pretrained(model_name).to(dev).eval()
    id2 = {int(k): v.lower() for k, v in m.config.id2label.items()}
    pos = next(i for i, lab in id2.items() if lab != "none")
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], truncation=True, padding=True, max_length=256,
                  return_tensors="pt").to(dev)
        out.append(torch.softmax(m(**enc).logits, dim=-1)[:, pos].cpu().numpy())
    del m
    if dev == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(out)


def main():
    merged = build_masked()
    print(f"Bảng masked: {len(merged)} câu unique | device={'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"{'='*70}")
    for p in PILLARS:
        nan_rows = merged[merged[p].isna()]
        cur_tot = int(merged[p].notna().sum())
        cur_pos = int((merged[p] == 1).sum())
        print(f"\n[{p.upper()}] hiện có {cur_tot} nhãn (pos={cur_pos}, "
              f"pos_rate={cur_pos/cur_tot:.2f}) | NaN cần gán: {len(nan_rows)}")
        pr = esgbert_pos_prob(nan_rows["text"].tolist(), ESGBERT[p])
        n_pos = int((pr >= TAU).sum())
        n_neg = int((pr <= 1 - TAU).sum())
        n_unc = len(pr) - n_pos - n_neg
        # cross-label chỉ áp train (~80%); ước lượng train-add
        print(f"  ESGBERT (τ={TAU}): +{n_pos} POSITIVE | +{n_neg} negative | {n_unc} uncertain(bỏ)")
        print(f"  => pos thêm vào (toàn bộ): +{n_pos} so với {cur_pos} hiện có "
              f"(+{100*n_pos/max(cur_pos,1):.0f}% positive) | train≈80% số này")
        print(f"  => tỉ lệ fill là positive: {100*n_pos/max(n_pos+n_neg,1):.0f}% "
              f"(còn lại là negative)")


if __name__ == "__main__":
    main()
