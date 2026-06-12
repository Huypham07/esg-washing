"""EDA gold + translate: phan bo nhan, do dai, overlap, align EN-VI.

  python scripts/data_report.py  -> outputs/metrics/data_report.json + in ra man hinh
"""
import json
import sys

import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from esgwash.data import claim_merge, topic_merge
from esgwash.data.gold_loader import load_action, load_claim_pair, load_ml_promise


def df_summary(df, label_cols):
    out = {"n": int(len(df)),
           "median_chars": float(df["text"].str.len().median()),
           "n_empty_text": int((df["text"].str.len() < 2).sum())}
    for c in label_cols:
        sub = df[c].dropna()
        out[f"pos_rate_{c}"] = round(float(sub.mean()), 4)
    return out


def main():
    report = {}
    # nguon topic 2k + env_claims da merge vao topic_masked.parquet (file goc o unused/)
    cp = load_claim_pair("vi")
    report["claim_pair"] = df_summary(cp, ["commitment", "specificity"])
    report["claim_pair"]["spec_missing_after_merge"] = int(cp["specificity"].isna().sum())
    report["action_500"] = df_summary(load_action("vi"), ["action"])
    try:
        mp = load_ml_promise("vi")
        report["ml_promise_vi"] = df_summary(mp, ["promise", "evidence"])
    except FileNotFoundError as e:
        report["ml_promise_vi"] = f"chua co: {e}"
        mp = load_ml_promise("en")
        report["ml_promise_src"] = df_summary(mp, ["promise", "evidence"])

    topic = pd.read_parquet("data/processed/gold/topic_masked.parquet")
    report["topic_masked"] = topic_merge.label_stats(topic)
    report["topic_masked"]["multi_pillar_rows"] = int(
        (topic[["env", "soc", "gov"]].notna().sum(axis=1) > 1).sum())
    claim = claim_merge.build_claim_table("vi")
    report["claim_table"] = claim_merge.label_stats(claim)

    out = Path("outputs/metrics/data_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
