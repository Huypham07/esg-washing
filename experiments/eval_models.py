"""Danh gia 2 model (topic, commitment) tren test — bao boc tach per-head + visualize.

Tai weights tu HF Hub (public), load qua MultiHeadTrainer, do tren file test phang.
Topic test = GOLD thuan (o non-gold = NaN -> tu mask). Claim test = climatebert chuan.

  python experiments/eval_models.py            # ca 2
  python experiments/eval_models.py --only topic
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from huggingface_hub import snapshot_download
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, precision_recall_fscore_support)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from esgwash.models.trainer import MultiHeadTrainer

# commitment = dqa2412 (model deploy trong pipeline: CommitmentHF, softmax 2 lop +
# underthesea); topic = MultiHeadTrainer in-repo. Dung DUNG model deploy de F1 khop ket qua.
REPOS = {"topic": "huypham71/esg-topic", "commitment": "dqa2412/esg-washing-optimized"}
TEST = {"topic": "data/topic_test.parquet", "commitment": "data/commitment_test.parquet"}
BASELINE = {"topic": 0.813, "commitment": 0.655}  # TF-IDF+LR macro-F1 (run_baselines)
OUT = Path("experiments/eval")


def _predict(name: str):
    """-> (heads, thresholds, probs_df, df). Commitment nap qua CommitmentHF (dqa2412,
    dung model deploy); topic qua MultiHeadTrainer. Text test la tho, ca hai loader tu tach tu."""
    df = pd.read_parquet(TEST[name])
    texts = df["text"].tolist()
    if name == "commitment":
        from esgwash.models.commitment_hf import CommitmentHF
        chf = CommitmentHF(repo=REPOS["commitment"])
        return ["commitment"], {"commitment": chf.threshold}, \
            pd.DataFrame({"commitment": chf.predict_proba(texts)}), df
    d = snapshot_download(REPOS[name])
    cfg = json.loads((Path(d) / "config.json").read_text(encoding="utf-8"))
    model = MultiHeadTrainer(cfg).load(d)
    return cfg["heads"], model.thresholds, model.predict_proba(texts), df


def eval_one(name: str) -> dict:
    heads, thr, probs, df = _predict(name)

    rows, f1s_tuned, f1s_half = [], [], []
    # squeeze=False: giu axes luon 2D ke ca khi 1 head (commitment) -> axes[i, j] khong vo.
    fig, axes = plt.subplots(2, len(heads), figsize=(4.5 * len(heads), 8), squeeze=False)
    for j, h in enumerate(heads):
        y = df[h].to_numpy(dtype=float)
        valid = ~np.isnan(y)
        yv = y[valid].astype(int)
        pv = probs[h].to_numpy()[valid]
        t = thr.get(h, 0.5)
        pred_t = (pv >= t).astype(int)
        pred_05 = (pv >= 0.5).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(yv, pred_t, average="binary",
                                                      zero_division=0)
        _, _, f1_05, _ = precision_recall_fscore_support(yv, pred_05, average="binary",
                                                         zero_division=0)
        ap = average_precision_score(yv, pv) if len(set(yv)) > 1 else float("nan")
        cm = confusion_matrix(yv, pred_t, labels=[0, 1])
        rows.append({"head": h, "n_eval": int(valid.sum()), "pos": int(yv.sum()),
                     "neg": int((yv == 0).sum()), "threshold": round(float(t), 2),
                     "precision": round(float(p), 4), "recall": round(float(r), 4),
                     "f1": round(float(f1), 4), "f1@0.5": round(float(f1_05), 4),
                     "avg_precision": round(float(ap), 4),
                     "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                     "fn": int(cm[1, 0]), "tp": int(cm[1, 1])})
        f1s_tuned.append(f1); f1s_half.append(f1_05)

        # hang 1: confusion matrix
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0, j],
                    xticklabels=["pred 0", "pred 1"], yticklabels=["true 0", "true 1"])
        axes[0, j].set_title(f"{h}  (thr={t:.2f})  F1={f1:.3f}")
        # hang 2: score distribution theo nhan that
        axes[1, j].hist(pv[yv == 0], bins=25, alpha=0.6, label="true 0", color="tab:blue")
        axes[1, j].hist(pv[yv == 1], bins=25, alpha=0.6, label="true 1", color="tab:red")
        axes[1, j].axvline(t, color="k", ls="--", lw=1, label=f"thr={t:.2f}")
        axes[1, j].set_title(f"{h}: phan bo score | AP={ap:.3f}")
        axes[1, j].legend(fontsize=8)
    fig.suptitle(f"{name.upper()} test — confusion + score distribution", y=1.0)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}_diagnostics.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    macro = round(float(np.mean(f1s_tuned)), 4)
    # bar so sanh F1 per-head vs baseline (khong ve macro)
    fig2, ax = plt.subplots(figsize=(1.4 * len(heads) + 2, 4))
    bars = ax.bar(heads, f1s_tuned, color="tab:green")
    ax.axhline(BASELINE[name], color="red", ls="--", label=f"baseline TF-IDF+LR={BASELINE[name]}")
    for b, v in zip(bars, f1s_tuned):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.05); ax.set_ylabel("F1"); ax.legend()
    ax.set_title(f"{name.upper()} — F1 per-head (test) vs baseline")
    fig2.tight_layout()
    fig2.savefig(OUT / f"{name}_f1_vs_baseline.png", dpi=120, bbox_inches="tight")
    plt.close(fig2)

    return {"model": name, "repo": REPOS[name], "macro_f1": macro,
            "macro_f1@0.5": round(float(np.mean(f1s_half)), 4),
            "baseline_macro_f1": BASELINE[name], "per_head": rows}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(REPOS), default=None)
    args = ap.parse_args(argv)
    names = [args.only] if args.only else list(REPOS)
    report = {n: eval_one(n) for n in names}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "eval_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
