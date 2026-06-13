"""Tune sieu tham so bang Optuna — TPE Bayesian + median pruning, khong grid search.

Nguyen tac:
- TPESampler multivariate: hoc phan phoi tham so tot tu cac trial truoc;
  lr/weight_decay sample tren thang log (dung scale cua tham so).
- MedianPruner: cat som trial co val macro-F1 per-epoch duoi median — tiet kiem
  ngan sach cho vung tham so hua hen (qua epoch_callback cua MultiHeadTrainer.fit).
- Study luu SQLite -> resumable (chay them trial khong mat lich su).
- Tune voi 1 seed co dinh tren val; KHONG tune epochs nhu mot chieu rieng:
  epochs = max_epochs co dinh, early-stop ngam qua best-epoch checkpoint trong fit.
- Tach biet tune (val) / danh gia cuoi (test, multi-seed) — khong cham test khi tune.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from esgwash.models.trainer import MultiHeadTrainer

# (low, high) — float; batch_size la categorical
DEFAULT_SPACE = {
    "lr": [1e-5, 5e-5],              # log
    "batch_size": [16, 32],
    "warmup_ratio": [0.0, 0.2],
    "weight_decay": [1e-3, 0.1],     # log
    "dropout": [0.05, 0.3],
}


def _suggest(trial, space: dict) -> dict:
    return {
        "lr": trial.suggest_float("lr", *space["lr"], log=True),
        "batch_size": trial.suggest_categorical("batch_size", list(space["batch_size"])),
        "warmup_ratio": trial.suggest_float("warmup_ratio", *space["warmup_ratio"]),
        "weight_decay": trial.suggest_float("weight_decay", *space["weight_decay"], log=True),
        "dropout": trial.suggest_float("dropout", *space["dropout"]),
    }


def tune(config: dict, train_df: pd.DataFrame, val_df: pd.DataFrame, *,
         n_trials: int, study_dir: str | Path, study_name: str,
         trainer_cls: type | None = None, text_col: str = "text", seed: int = 42):
    """Chay/tiep tuc study Optuna, tra ve study (best qua study.best_params)."""
    import optuna

    trainer_cls = trainer_cls or MultiHeadTrainer
    tune_cfg = config.get("tune", {})
    space = {**DEFAULT_SPACE, **tune_cfg.get("space", {})}
    max_epochs = int(tune_cfg.get("max_epochs", 10))

    study_dir = Path(study_dir)
    study_dir.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{study_dir / 'optuna.db'}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))

    def objective(trial):
        params = _suggest(trial, space)
        cfg = {**config, "dropout": params["dropout"],
               "train": {**config["train"],
                         **{k: v for k, v in params.items() if k != "dropout"},
                         "epochs": max_epochs}}
        trainer = trainer_cls(cfg)

        def on_epoch(epoch, val_metrics):
            trial.report(val_metrics["macro_f1"], step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        info = trainer.fit(train_df, val_df, text_col=text_col, seed=seed,
                           epoch_callback=on_epoch)
        return info["best_val_macro_f1"]

    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    return study


def save_best_params(study, path: str | Path, max_epochs: int) -> dict:
    best = {"params": study.best_params,
            "val_macro_f1": round(float(study.best_value), 4),
            "epochs": max_epochs,
            "n_trials": len(study.trials),
            "tuned_at": datetime.now().isoformat(timespec="seconds")}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(best, indent=2), encoding="utf-8")
    return best


def apply_best_params(config: dict, best: dict) -> dict:
    """Ghi best params vao config train cuoi; epochs = max_epochs cua tune
    (best-epoch checkpoint trong fit lo early stopping)."""
    p = best["params"]
    return {**config, "dropout": p["dropout"],
            "train": {**config["train"],
                      "lr": p["lr"], "batch_size": p["batch_size"],
                      "warmup_ratio": p["warmup_ratio"],
                      "weight_decay": p["weight_decay"],
                      "epochs": int(best.get("epochs", config["train"]["epochs"]))}}
