"""Dump TOÀN BỘ trial của 5 study Optuna (gold) để review tuning.

Chạy:  python outputs/_review_tuning_results.py
Đọc:   esg_<task>_hyperopt.db (sqlite) — không train lại gì.
"""
import os
import sys
import statistics
from pathlib import Path

import optuna

sys.stdout.reconfigure(encoding="utf-8")
optuna.logging.set_verbosity(optuna.logging.WARNING)
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

TASKS = ["env", "soc", "gov", "commitment", "specificity"]

for task in TASKS:
    name = f"esg_{task}_hyperopt"
    db = ROOT / f"{name}.db"
    if not db.exists():
        print(f"[{task}] MISSING {db}\n")
        continue

    study = optuna.load_study(study_name=name, storage=f"sqlite:///{name}.db")
    trials = sorted(study.trials, key=lambda t: t.number)

    print("=" * 80)
    print(f"TASK = {task}   (study {name},  {len(trials)} trial)")
    print("-" * 80)
    print(f"{'#':>2} {'state':<9} {'lr':>9} {'batch':>5} {'wd':>7} {'ep':>3} {'val_macroF1':>11} {'sec':>5}")

    best_num = study.best_trial.number if study.best_trial else None
    vals = []
    for t in trials:
        p = t.params
        lr = p.get("learning_rate")
        bs = p.get("train_batch_size")
        wd = p.get("weight_decay")
        ep = p.get("epochs")
        lr_s = f"{lr:.2e}" if lr is not None else "-"
        wd_s = f"{wd:.4f}" if wd is not None else "-"
        val_s = f"{t.value:.4f}" if t.value is not None else "(none)"
        dur = f"{t.duration.total_seconds():.0f}" if t.duration else "-"
        mark = "  <== BEST" if t.number == best_num else ""
        print(f"{t.number:>2} {t.state.name:<9} {lr_s:>9} {str(bs):>5} {wd_s:>7} {str(ep):>3} {val_s:>11} {dur:>5}{mark}")
        if t.value is not None:
            vals.append(t.value)

    if vals:
        print("-" * 80)
        print(f"best={max(vals):.4f}  worst={min(vals):.4f}  spread={max(vals)-min(vals):.4f}  "
              f"mean={statistics.mean(vals):.4f}  std={statistics.pstdev(vals):.4f}  n_done={len(vals)}")
        print(f"BEST params -> {study.best_trial.params}")
    print()
