import json
import copy
from pathlib import Path

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from src.training.train_model import (
    load_yaml_config,
    resolve_runtime_config,
    train_once,
)


def objective(trial: optuna.Trial, base_config: dict) -> float:
    config = copy.deepcopy(base_config)

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    batch_size = trial.suggest_categorical("train_batch_size", [8, 16, 32])
    max_length = trial.suggest_categorical("max_length", [128, 256])
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.10, step=0.0001)
    epochs = trial.suggest_int("epochs", 3, 10)

    config["training"]["learning_rate"] = learning_rate
    config["training"]["train_batch_size"] = batch_size
    config["training"]["weight_decay"] = weight_decay
    config["training"]["epochs"] = epochs
    config["model"]["max_length"] = max_length

    if "neuro_symbolic" in config:
        constraint_lambda = trial.suggest_float("constraint_lambda", 0.05, 0.70, step=0.0001)
        config["neuro_symbolic"]["constraint_lambda"] = constraint_lambda

    config["paths"]["output_dir"] = str(
        Path(config["paths"]["output_dir"]) / f"trial_{trial.number}"
    )

    lambda_str = f", λ: {config['neuro_symbolic']['constraint_lambda']:.4f}" if "neuro_symbolic" in config else ""
    print(f"\n{'='*50}")
    print(f"TRIAL {trial.number}")
    print(
        f"LR: {learning_rate:.4e}, Batch: {batch_size}, MaxLen: {max_length}, "
        f"WD: {weight_decay:.4f}, Epochs: {epochs}{lambda_str}"
    )
    print(f"{'='*50}")

    run_results = train_once(config, save_model=False)
    macro_f1 = run_results["trainer"].state.best_metric
    print(f"Trial {trial.number} → Macro-F1: {macro_f1:.4f}")
    return macro_f1


def main(args_cli=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/train.yml")
    parser.add_argument("--task", type=str, default="topic", choices=["topic", "action"])
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args(args_cli)

    raw_config = load_yaml_config(Path(args.config))
    base_config = resolve_runtime_config(raw_config, task=args.task)

    study_name = f"esg_{args.task}_hyperopt"

    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{study_name}.db",
        direction="maximize",
        sampler=TPESampler(seed=base_config.get("seed", 42)),
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
