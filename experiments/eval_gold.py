"""Danh gia pipeline (topic/commitment/specificity) tren bo gold 400 chunk,
so voi TUNG nguoi gan nhan (A, B) + tran nguoi-nguoi.

Y tuong (Muc reliability cua paper): khong co mot ground-truth tuyet doi vi 2
nguoi gan bat dong (commit kappa=0.18). Thay vi ep ve 1 gold, ta do model so voi
A va so voi B rieng, roi dat canh tran A-B. Neu model nam TRONG dai bat dong
nguoi-nguoi (kappa(model,*) ~ kappa(A,B)) thi model tin cay ngang mot con nguoi.

Can GPU (PhoBERT + Qwen3 LLM) -> chay tren Kaggle. Nap model qua run.load_models
(tai tu HF Hub), classify toan bo 400 chunk roi cham diem.

  python experiments/eval_gold.py            # full (specificity tren ca 400 chunk)
  python experiments/eval_gold.py --limit 40 # smoke test
  python experiments/eval_gold.py --no-spec  # bo qua LLM specificity (chi topic/commit)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from esgwash.run import classify_chunks, load_models  # noqa: E402

GOLD = {"A": ROOT / "data/gold_annot_1_relabeled.xlsx",
        "B": ROOT / "data/gold_annot_2_relabeled.xlsx"}
SHEET = "Sheet1"
OUT = ROOT / "experiments/eval"
# (ten hien thi, cot gold, cot du doan model)
BIN = [("env", "g_env", "is_env"), ("soc", "g_soc", "is_soc"),
       ("gov", "g_gov", "is_gov"), ("commit", "g_is_commit", "is_commitment"),
       # 5 atomic flags — gold col name == model output col name
       ("co_cam_ket", "co_cam_ket", "co_cam_ket"),
       ("co_hanh_dong_ten", "co_hanh_dong_ten", "co_hanh_dong_ten"),
       ("co_so_dinh_luong", "co_so_dinh_luong", "co_so_dinh_luong"),
       ("quy_ve_bank", "quy_ve_bank", "quy_ve_bank"),
       ("co_moc_tg", "co_moc_tg", "co_moc_tg")]


def load_gold() -> pd.DataFrame:
    """Merge A/B theo chunk_id; giu content_text + nhan moi nguoi (suffix _A/_B)."""
    a = pd.read_excel(GOLD["A"], sheet_name=SHEET)
    b = pd.read_excel(GOLD["B"], sheet_name=SHEET)
    atomic_flags = ["co_cam_ket", "co_hanh_dong_ten", "co_so_dinh_luong",
                    "quy_ve_bank", "co_moc_tg"]
    keep = ["chunk_id", "g_env", "g_soc", "g_gov", "g_is_commit", "g_spec_level"] + atomic_flags
    m = a[["chunk_id", "content_text"] + keep[1:]].merge(
        b[keep], on="chunk_id", suffixes=("_A", "_B"))
    m["content_text"] = m["content_text"].astype(str)
    return m


def _binary_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """F1 + accuracy + Cohen kappa tren cac o ca hai deu co nhan."""
    keep = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[keep].astype(int), y_pred[keep].astype(int)
    if len(yt) == 0:
        return {"n": 0}
    p, r, f1, _ = precision_recall_fscore_support(yt, yp, average="binary", zero_division=0)
    kap = cohen_kappa_score(yt, yp) if len(set(yt)) > 1 and len(set(yp)) > 1 else float("nan")
    return {"n": int(len(yt)), "precision": round(float(p), 4), "recall": round(float(r), 4),
            "f1": round(float(f1), 4), "accuracy": round(float((yt == yp).mean()), 4),
            "cohen_kappa": round(float(kap), 4)}


def _spec_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """QWK + accuracy cho thang ordinal; + phan biet nhi phan QDR (muc==2 vs con lai)."""
    keep = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[keep].astype(int), y_pred[keep].astype(int)
    if len(yt) == 0:
        return {"n": 0}
    qwk = cohen_kappa_score(yt, yp, weights="quadratic") if len(set(yt)) > 1 else float("nan")
    return {"n": int(len(yt)), "accuracy": round(float((yt == yp).mean()), 4),
            "quadratic_weighted_kappa": round(float(qwk), 4),
            "qdr_kappa": _binary_scores((yt == 2).astype(float),
                                        (yp == 2).astype(float)).get("cohen_kappa"),
            "rate_pred": {int(k): round(float((yp == k).mean()), 3) for k in (0, 1, 2)},
            "rate_true": {int(k): round(float((yt == k).mean()), 3) for k in (0, 1, 2)}}


def evaluate(clf: pd.DataFrame, gold: pd.DataFrame, with_spec: bool) -> dict:
    """clf da co cot du doan; gold co nhan _A/_B. So model<->A, model<->B, A<->B (tran)."""
    m = gold.merge(clf, on="chunk_id", suffixes=("", "_clf"))
    report = {"n_chunks": int(len(m)), "labels": {}}
    for name, gcol, pcol in BIN:
        pred = m[pcol].to_numpy(dtype=float)
        ya, yb = m[f"{gcol}_A"].to_numpy(dtype=float), m[f"{gcol}_B"].to_numpy(dtype=float)
        report["labels"][name] = {
            "model_vs_A": _binary_scores(ya, pred),
            "model_vs_B": _binary_scores(yb, pred),
            "human_ceiling_A_vs_B": _binary_scores(ya, yb)}
    if with_spec and "spec_level" in m:
        pred = m["spec_level"].to_numpy(dtype=float)
        ya, yb = m["g_spec_level_A"].to_numpy(dtype=float), m["g_spec_level_B"].to_numpy(dtype=float)
        report["labels"]["spec_level"] = {
            "model_vs_A": _spec_scores(ya, pred),
            "model_vs_B": _spec_scores(yb, pred),
            "human_ceiling_A_vs_B": _spec_scores(ya, yb)}
    return report


def plot_within_band(report: dict) -> None:
    """Bar kappa: model-vs-A, model-vs-B, va tran A-vs-B cho moi nhan."""
    names, mA, mB, ceil = [], [], [], []
    for name, d in report["labels"].items():
        key = ("quadratic_weighted_kappa" if name == "spec_level" else "cohen_kappa")
        names.append(name + ("\n(QWK)" if name == "spec_level" else ""))
        mA.append(d["model_vs_A"].get(key, np.nan))
        mB.append(d["model_vs_B"].get(key, np.nan))
        ceil.append(d["human_ceiling_A_vs_B"].get(key, np.nan))
    x = np.arange(len(names)); w = 0.27
    fig, ax = plt.subplots(figsize=(1.7 * len(names) + 2, 4.5))
    ax.bar(x - w, mA, w, label="model vs A", color="tab:blue")
    ax.bar(x, mB, w, label="model vs B", color="tab:cyan")
    ax.bar(x + w, ceil, w, label="human ceiling (A vs B)", color="tab:grey", hatch="//")
    for xi, c in zip(x, ceil):
        if not np.isnan(c):
            ax.axhline(c, xmin=(xi + 0.5) / len(names) - 0.13, xmax=(xi + 0.5) / len(names) + 0.13,
                       color="black", lw=0.8, ls=":")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("kappa"); ax.set_ylim(0, 1.0)
    ax.set_title("Model vs each annotator vs human ceiling\n(model within the A-B band = as reliable as a human)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "eval_gold_within_band.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0=full; >0 = N chunk dau (smoke test)")
    ap.add_argument("--no-spec", action="store_true", help="bo qua LLM specificity")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    gold = load_gold()
    if args.limit:
        gold = gold.head(args.limit).copy()
    print(f"### eval gold: {len(gold)} chunk")

    models = load_models()
    spec_model = None if args.no_spec else models["specificity"]
    # spec_on_commitment=False: cham specificity tren CA 400 chunk de co the so model
    # spec_level voi TUNG annotator (tranh confound bang cong commitment cua model).
    clf = classify_chunks(gold[["chunk_id", "content_text"]], models["topic"],
                          models["commitment"], spec_model, spec_on_commitment=False)

    report = evaluate(clf, gold, with_spec=not args.no_spec)
    OUT.mkdir(parents=True, exist_ok=True)
    clf.to_parquet(OUT / "gold_classified.parquet", index=False)
    (OUT / "eval_gold_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_within_band(report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"-> {OUT}/eval_gold_report.json, gold_classified.parquet, eval_gold_within_band.png")


if __name__ == "__main__":
    main()
