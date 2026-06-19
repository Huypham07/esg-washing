"""Phase 3 (CPU): interpretable lexical baseline + cross-lingual transfer table.
Transformer/encoder comparison + transformer SHAP run on Kaggle (GPU/auth)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from esgwash.eda.style import PALETTE, SCORE_CMAP, apply_rcparams, style_axes
from esgwash.models.interpret import cross_lingual_macro_f1, top_tokens

_CONFIGS = [("VI->VI", "text", "text"), ("EN->VI", "text_en", "text"),
            ("EN->EN", "text_en", "text_en")]


def transfer_table(train_df, test_df, heads) -> pd.DataFrame:
    rows = []
    for name, tr_col, te_col in _CONFIGS:
        f1 = cross_lingual_macro_f1(train_df, test_df, heads, tr_col, te_col)
        rows.append({"config": name, "macro_f1": round(f1, 4)})
    return pd.DataFrame(rows)


def _fig_tokens(train_df, heads, title, out_fig: Path, fname: str) -> Path:
    fig, axes = plt.subplots(1, len(heads), figsize=(5 * len(heads), 5),
                             facecolor=PALETTE["paper"])
    if len(heads) == 1:
        axes = [axes]
    for ax, h in zip(axes, heads):
        toks = top_tokens(train_df, h, "text", k=12)[::-1]
        labels = [t for t, _ in toks]
        vals = [w for _, w in toks]
        style_axes(ax, f"{h}", "Top TF-IDF+LR tokens (Vietnamese).")
        ax.barh(range(len(labels)), vals,
                color=SCORE_CMAP(np.linspace(0.2, 0.95, len(labels))),
                edgecolor=PALETTE["paper"])
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.set_xlabel("LR coefficient")
    fig.suptitle(title, x=0.02, ha="left", fontweight="bold", color=PALETTE["ink"])
    fig.tight_layout()
    out_fig.mkdir(parents=True, exist_ok=True)
    p = out_fig / fname
    fig.savefig(p, bbox_inches="tight", facecolor=PALETTE["paper"])
    plt.close(fig)
    return p


def main(out_fig: str = "experiments/figures", out_tab: str = "experiments/eval") -> dict:
    apply_rcparams()
    topic_tr = pd.read_parquet("data/topic_train.parquet")
    topic_te = pd.read_parquet("data/topic_test.parquet")
    com_tr = pd.read_parquet("data/commitment_train.parquet")
    com_te = pd.read_parquet("data/commitment_test.parquet")
    topic_heads = [h for h in ["env", "soc", "gov"] if h in topic_tr.columns]

    t_topic = transfer_table(topic_tr, topic_te, topic_heads).assign(task="topic")
    t_com = transfer_table(com_tr, com_te, ["commitment"]).assign(task="commitment")
    transfer = pd.concat([t_topic, t_com], ignore_index=True)

    tab = Path(out_tab); tab.mkdir(parents=True, exist_ok=True)
    transfer.to_csv(tab / "transfer.csv", index=False)

    tok_rows = []
    for task, df, heads in [("topic", topic_tr, topic_heads), ("commitment", com_tr, ["commitment"])]:
        for h in heads:
            for tok, w in top_tokens(df, h, "text", k=15):
                tok_rows.append({"task": task, "head": h, "token": tok, "coef": round(w, 4)})
    pd.DataFrame(tok_rows).to_csv(tab / "top_tokens.csv", index=False)

    fig = Path(out_fig)
    _fig_tokens(topic_tr, topic_heads, "Topic classifier — top tokens per pillar", fig, "tokens_topic.png")
    _fig_tokens(com_tr, ["commitment"], "Commitment classifier — top tokens", fig, "tokens_commitment.png")

    print("Cross-lingual transfer (TF-IDF+LR macro-F1):")
    print(transfer.to_string(index=False))
    return {"transfer": transfer}


if __name__ == "__main__":
    main()
