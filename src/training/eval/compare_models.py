"""So sánh bake-off (Phase 02b): gom metric -> bảng + biểu đồ (nhãn ENGLISH).

Nguồn (bỏ qua phần chưa có):
- Baseline: outputs/baselines/<task>_baseline.json  (majority, tfidf_lr)
- Model:    outputs/models/<dir>/metrics_summary.json (PhoBERT, Qwen, ...)

#params + infer câu/s = báo cáo thủ công (đã biết: PhoBERT ~135M, Qwen2.5-7B ~7.6B);
đo throughput là việc GPU local, không tự động hoá ở đây (YAGNI).

Chạy:  python -m src.training.eval.compare_models
       -> outputs/compare/bakeoff_table.csv + figures/phase02b/<metric>_by_task.png
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

MODELS_DIR = Path("outputs/models")
BASELINE_DIR = Path("outputs/baselines")
OUT_DIR = Path("outputs/compare")
FIG_DIR = Path("figures/phase02b")
TOPICS = ["env", "soc", "gov"]

# (label hiển thị) -> {topic: thư mục trong outputs/models}. Bỏ qua nếu chưa train.
MODEL_DIRS = {
    "PhoBERT": {"env": "topic_e", "soc": "topic_s", "gov": "topic_g"},
    "Qwen2.5-7B": {"env": "qwen_e", "soc": "qwen_s", "gov": "qwen_g"},
    "PhoGPT-4B": {"env": "phogpt_e", "soc": "phogpt_s", "gov": "phogpt_g"},
    "Vistral-7B": {"env": "vistral_e", "soc": "vistral_s", "gov": "vistral_g"},
}


def _read_model_metrics(models_dir: Path, label: str, topic_dirs: dict) -> list[dict]:
    rows = []
    for topic, d in topic_dirs.items():
        p = models_dir / d / "metrics_summary.json"
        if not p.exists():
            continue
        test = json.loads(p.read_text(encoding="utf-8")).get("test", {})
        rows.append({"model": label, "task": topic,
                     "macro_f1": test.get("macro_f1"), "f1_positive": test.get("f1_positive")})
    return rows


def _read_baselines(baseline_dir: Path) -> list[dict]:
    rows = []
    for topic in TOPICS:
        p = baseline_dir / f"{topic}_baseline.json"
        if not p.exists():
            continue
        res = json.loads(p.read_text(encoding="utf-8"))
        for key, label in (("majority", "Majority"), ("tfidf_lr", "TF-IDF+LR")):
            s = res.get(key, {})
            rows.append({"model": label, "task": topic,
                         "macro_f1": s.get("macro_f1"), "f1_positive": s.get("f1_positive")})
    return rows


def build_table(models_dir: Path = MODELS_DIR, baseline_dir: Path = BASELINE_DIR,
                out_dir: Path = OUT_DIR) -> pd.DataFrame:
    """Gom baseline + model -> DataFrame long [model, task, macro_f1, f1_positive] + lưu CSV."""
    rows = _read_baselines(Path(baseline_dir))
    for label, topic_dirs in MODEL_DIRS.items():
        rows.extend(_read_model_metrics(Path(models_dir), label, topic_dirs))
    df = pd.DataFrame(rows)
    if df.empty:
        print("Chưa có metric nào (train model / chạy baseline trước).")
        return df
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "bakeoff_table.csv", index=False)
    return df


def plot_comparison(df: pd.DataFrame, metric: str = "macro_f1", fig_dir: Path = FIG_DIR) -> Path | None:
    """Bar chart metric theo topic × model (nhãn English) -> PNG."""
    if df.empty:
        return None
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    pivot = df.pivot(index="task", columns="model", values=metric)
    ax = pivot.plot(kind="bar", figsize=(9, 5))
    ax.set_xlabel("ESG topic")
    ax.set_ylabel(metric.replace("_", "-").title())
    ax.set_title(f"Bake-off: {metric} by topic (vi_gold test)")
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f"{metric}_by_task.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def confusion_from_preds(output_dir: str | Path) -> pd.DataFrame:
    """Confusion matrix 2x2 từ <output_dir>/test_predictions.parquet (train_model đã lưu)."""
    from sklearn.metrics import confusion_matrix
    p = Path(output_dir) / "test_predictions.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Chưa có preds: {p} (chạy train_model trước).")
    d = pd.read_parquet(p)
    cm = confusion_matrix(d["y_true"], d["y_pred"], labels=[0, 1])
    return pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"])


def main(args=None) -> None:
    parser = argparse.ArgumentParser(description="So sánh bake-off PhoBERT vs LLM")
    parser.add_argument("--models-dir", type=str, default=str(MODELS_DIR))
    parser.add_argument("--baseline-dir", type=str, default=str(BASELINE_DIR))
    args = parser.parse_args(args)

    df = build_table(Path(args.models_dir), Path(args.baseline_dir))
    if df.empty:
        return
    print("=== Bake-off macro-F1 (vi_gold test) ===")
    print(df.pivot(index="task", columns="model", values="macro_f1").to_string())
    for metric in ("macro_f1", "f1_positive"):
        out = plot_comparison(df, metric)
        if out:
            print(f"Figure -> {out}")
    print(f"Table -> {OUT_DIR}/bakeoff_table.csv")


if __name__ == "__main__":
    main()
