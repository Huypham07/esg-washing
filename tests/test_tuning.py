"""Test wiring tune Optuna voi stub trainer (khong can backbone/transformers)."""
import json

import pandas as pd
import pytest

optuna = pytest.importorskip("optuna")

from esgwash.models.tuning import apply_best_params, save_best_params, tune


class StubTrainer:
    """Gia lap MultiHeadTrainer: diem = ham loi cua lr quanh 3e-5, co goi epoch_callback."""

    def __init__(self, config):
        self.config = config

    def fit(self, train_df, val_df, text_col="text", seed=42, epoch_callback=None):
        tc = self.config["train"]
        score = 1.0 - abs(float(tc["lr"]) - 3e-5) / 3e-5 - self.config["dropout"] * 0.1
        for epoch in range(tc["epochs"]):
            if epoch_callback is not None:
                epoch_callback(epoch, {"macro_f1": score * (epoch + 1) / tc["epochs"]})
        return {"best_val_macro_f1": score}


CONFIG = {
    "backbone": "stub", "labels": ["env", "soc", "gov"],
    "train": {"lr": 2e-5, "batch_size": 32, "epochs": 5,
              "warmup_ratio": 0.1, "weight_decay": 0.01, "seeds": [42]},
    "tune": {"max_epochs": 3},
}
DF = pd.DataFrame({"text": ["a", "b"], "env": [0, 1], "soc": [1, 0], "gov": [0, 0]})


def test_tune_save_apply(tmp_path):
    study = tune(CONFIG, DF, DF, n_trials=8, study_dir=tmp_path,
                 study_name="t", trainer_cls=StubTrainer, seed=42)
    assert len(study.trials) == 8
    assert (tmp_path / "optuna.db").exists()

    best = save_best_params(study, tmp_path / "best_params.json", max_epochs=3)
    saved = json.loads((tmp_path / "best_params.json").read_text())
    assert saved["params"] == study.best_params
    assert saved["epochs"] == 3

    cfg = apply_best_params(CONFIG, best)
    assert cfg["train"]["lr"] == study.best_params["lr"]
    assert cfg["train"]["epochs"] == 3
    assert cfg["dropout"] == study.best_params["dropout"]
    assert CONFIG["train"]["lr"] == 2e-5  # khong mutate config goc


def test_tune_resumable(tmp_path):
    tune(CONFIG, DF, DF, n_trials=3, study_dir=tmp_path,
         study_name="t", trainer_cls=StubTrainer, seed=42)
    study = tune(CONFIG, DF, DF, n_trials=2, study_dir=tmp_path,
                 study_name="t", trainer_cls=StubTrainer, seed=42)
    assert len(study.trials) == 5
