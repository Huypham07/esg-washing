"""EDA gold + translate: phan bo nhan, do dai, overlap, align EN-VI.

  python experiments/data_report.py  -> experiments/metrics/data_report.json + in ra man hinh
"""
import json
import sys

import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from esgwash.data import commitment_merge, topic_merge
from esgwash.data.gold_loader import load_action, load_commitment, load_ml_promise


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
    report["commitment_src"] = df_summary(load_commitment("vi"), ["commitment"])
    report["action_500"] = df_summary(load_action("vi"), ["action"])
    try:
        mp = load_ml_promise("vi")
        report["ml_promise_vi"] = df_summary(mp, ["promise", "evidence"])
    except FileNotFoundError as e:
        report["ml_promise_vi"] = f"chua co: {e}"
        mp = load_ml_promise("en")
        report["ml_promise_src"] = df_summary(mp, ["promise", "evidence"])

    topic = pd.concat([pd.read_parquet("data/topic_train.parquet"),
                       pd.read_parquet("data/topic_test.parquet")], ignore_index=True)
    report["topic"] = topic_merge.label_stats(topic)
    report["topic"]["multi_pillar_rows"] = int(
        (topic[["env", "soc", "gov"]].notna().sum(axis=1) > 1).sum())
    claim = commitment_merge.build_commitment_table("vi")
    report["commitment_table"] = commitment_merge.label_stats(claim)

    out = Path("experiments/metrics/data_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
