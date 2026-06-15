"""Huấn luyện topic / commitment: Optuna tune -> train multi-seed.

  python -m esgwash.train topic --tune 30        # tune 30 trial (resumable) rồi train
  python -m esgwash.train topic                  # train bằng hyperparams trong config
  python -m esgwash.train commitment

Data phẳng: data/<task>_{train,test}.parquet (val cắt từ train lúc chạy). Model ->
outputs/models/<task>/<variant>/; metrics -> experiments/metrics/<task>_<variant>.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from esgwash.config import load_config
from esgwash.data.topic_merge import PILLARS, carve_val
from esgwash.models.commitment_model import CommitmentModel
from esgwash.models.topic_model import TopicModel
from esgwash.models.trainer import multi_seed
from esgwash.models.tuning import apply_best_params, save_best_params, tune

DATA = Path("data")
METRICS = Path("experiments/metrics")


def _tune_then_train(cfg, train, val, test, *, study_name, trainer_cls, n_trials, out_dir):
    # Khong --tune: dung hyperparams trong config (da tuned). --tune N: tim lai, ap dung cho
    # lan train nay + ghi best_params.json; copy ket qua vao config de dat lam mac dinh.
    best = None
    if n_trials > 0:
        study = tune(cfg, train, val, n_trials=n_trials, study_dir=out_dir,
                     study_name=study_name, trainer_cls=trainer_cls)
        best = save_best_params(study, out_dir / "best_params.json",
                                max_epochs=int(cfg.get("tune", {}).get("max_epochs", 10)))
        cfg = apply_best_params(cfg, best)
        print(f"tune xong: val macro-F1 {best['val_macro_f1']} -> copy params vao config")
    res = multi_seed(cfg, train, val, test, out_dir=out_dir, trainer_cls=trainer_cls)
    res["tuned_params"] = best
    return res


def _load_split(task: str, heads: list[str], lang: str, mask_col: str | None = None):
    train_full = pd.read_parquet(DATA / f"{task}_train.parquet")
    test = pd.read_parquet(DATA / f"{task}_test.parquet")
    if lang == "en":
        train_full = train_full.assign(text=train_full["text_en"])
        test = test.assign(text=test["text_en"])
    mask = (train_full[mask_col[0]] == mask_col[1]).to_numpy() if mask_col else None
    train, val = carve_val(train_full, heads, seed=42, mask=mask)
    return train, val, test


def _write_metrics(name: str, res: dict) -> None:
    METRICS.mkdir(parents=True, exist_ok=True)
    (METRICS / f"{name}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res.get("aggregate"), indent=2))


def train_topic(lang: str = "vi", n_trials: int = 0) -> dict:
    cfg = load_config("topic")
    cfg["heads"] = cfg["labels"]
    if lang == "en":
        cfg["backbone"] = cfg.get("en_backbone", "roberta-base"); cfg["word_segment"] = False
    train, val, test = _load_split("topic", PILLARS, lang)
    res = _tune_then_train(cfg, train, val, test, study_name=f"topic_{lang}",
                           trainer_cls=TopicModel, n_trials=n_trials,
                           out_dir=Path(cfg["output_dir"]) / lang)
    _write_metrics(f"topic_{lang}", res)
    return res


def train_commitment(lang: str = "vi", n_trials: int = 0) -> dict:
    cfg = load_config("commitment_train")
    cfg["heads"] = ["commitment"]
    if lang == "en":
        cfg["backbone"] = cfg.get("en_backbone", "roberta-base"); cfg["word_segment"] = False
    train, val, test = _load_split("commitment", ["commitment"], lang,
                                   mask_col=("source", "climatebert"))
    res = _tune_then_train(cfg, train, val, test, study_name=f"commitment_{lang}",
                           trainer_cls=CommitmentModel, n_trials=n_trials,
                           out_dir=Path(cfg["output_dir"]) / lang)
    _write_metrics(f"commitment_{lang}", res)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["topic", "commitment"])
    ap.add_argument("--lang", default="vi", choices=["vi", "en"])
    ap.add_argument("--tune", type=int, default=0, metavar="N")
    args = ap.parse_args(argv)
    (train_topic if args.task == "topic" else train_commitment)(lang=args.lang, n_trials=args.tune)


if __name__ == "__main__":
    main()
