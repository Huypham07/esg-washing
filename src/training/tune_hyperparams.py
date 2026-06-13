import json
import copy
import shutil
from pathlib import Path

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from transformers import TrainerCallback

from src.training.train_model import (
    load_yaml_config,
    resolve_runtime_config,
    train_once,
)


class OptunaPruningCallback(TrainerCallback):
    """Report dev-metric mỗi lần eval -> MedianPruner cắt sớm trial tệ (Phase 02).
    Pruner cũ vô dụng vì objective chỉ trả best cuối, KHÔNG report intermediate."""

    def __init__(self, trial, metric_name: str = "eval_macro_f1"):
        self.trial = trial
        self.metric_name = metric_name
        self.step = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or self.metric_name not in metrics:
            return
        self.trial.report(metrics[self.metric_name], step=self.step)
        self.step += 1
        if self.trial.should_prune():
            raise optuna.TrialPruned()


def objective(trial: optuna.Trial, base_config: dict) -> float:
    config = copy.deepcopy(base_config)
    model_type = config["model"].get("type", "encoder")

    if model_type == "decoder_lora":
        # LEAN: mỗi trial = train LLM 7B (~20-30 phút) -> không sweep batch/max_length.
        config["training"]["learning_rate"] = trial.suggest_float("learning_rate", 5e-5, 3e-4, log=True)
        config["training"]["epochs"] = trial.suggest_int("epochs", 2, 4)
        lora_r = trial.suggest_categorical("lora_r", [8, 16, 32])
        config["model"].setdefault("lora", {})
        config["model"]["lora"]["r"] = lora_r
        config["model"]["lora"]["alpha"] = 2 * lora_r
    else:
        # encoder (PhoBERT) — search LR/batch/wd/warmup (Phase 02: BỎ epochs, THÊM warmup).
        # max_length KHÔNG search: set per-task từ data (topic=128, subst=256) trong config/train.yml.
        config["training"]["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
        # batch search per-task từ config (subst data ít -> [8,16]; topic -> [8,16,32]). Xem config/train.yml.
        batch_choices = config.get("tune", {}).get("train_batch_size", [8, 16, 32])
        config["training"]["train_batch_size"] = trial.suggest_categorical("train_batch_size", batch_choices)
        config["training"]["weight_decay"] = trial.suggest_float("weight_decay", 0.0, 0.10, step=0.0001)
        # warmup_ratio: lever ổn định fine-tune (Mosbach/Dodge). KHÔNG tune dropout (chốt 2026-06-13).
        config["training"]["warmup_ratio"] = trial.suggest_float("warmup_ratio", 0.0, 0.2)
        # epochs KHÔNG tune: cố định = trần config (10) + EarlyStopping quyết best-epoch.

    trial_dir = Path(config["paths"]["output_dir"]) / f"trial_{trial.number}"
    config["paths"]["output_dir"] = str(trial_dir)

    # Không lưu checkpoint trong quá trình tuning để tiết kiệm disk.
    # EarlyStoppingCallback vẫn track best_metric qua state nên metric vẫn đúng.
    config["training"]["save_strategy"] = "no"
    config["training"]["load_best_model_at_end"] = False

    print(f"\n{'='*50}")
    print(f"TRIAL {trial.number} [{model_type}] — {trial.params}")
    print(f"{'='*50}")

    metric_name = "eval_" + config["training"].get("metric_for_best_model", "macro_f1")
    pruning_cb = OptunaPruningCallback(trial, metric_name=metric_name)
    # try/finally: trial bị PRUNE sẽ raise TrialPruned giữa chừng -> phải dọn trial_dir ở finally
    # (nếu không, mỗi trial pruned để lại 1 thư mục trial_N rỗng trong outputs/models/<dir>/).
    try:
        run_results = train_once(config, save_model=False, extra_callbacks=[pruning_cb])
        macro_f1 = run_results["trainer"].state.best_metric
        print(f"Trial {trial.number} -> Macro-F1: {macro_f1:.4f}")
        return macro_f1
    finally:
        shutil.rmtree(trial_dir, ignore_errors=True)


def main(args_cli=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/train.yml")
    parser.add_argument("--task", type=str, default="topic")
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args(args_cli)

    raw_config = load_yaml_config(Path(args.config))
    base_config = resolve_runtime_config(raw_config, task=args.task)

    study_name = f"esg_{args.task}_hyperopt"

    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{study_name}.db",
        direction="maximize",
        # n_startup_trials=5: với n_trials=10 -> 5 random thăm dò + 5 TPE khai thác.
        # (Mặc định 10 sẽ khiến cả 10 trial là random startup, TPE không bao giờ bật.)
        sampler=TPESampler(seed=base_config.get("seed", 42), n_startup_trials=5),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0),
        load_if_exists=True,
    )

    print(f"Optuna tuning: task={args.task}, trials={args.trials}")
    study.optimize(
        lambda trial: objective(trial, base_config),
        n_trials=args.trials,
    )

    best = study.best_trial
    print(f"\nBest trial #{best.number} — Macro-F1: {best.value:.4f}")
    print("Params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    out_path = Path(base_config["paths"]["output_dir"]) / f"best_params_{args.task}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(best.params, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
