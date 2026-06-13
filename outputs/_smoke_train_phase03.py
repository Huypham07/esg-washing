"""Smoke-test TRAINING path (CÁCH C — ngưỡng 0.5) trên SUBST task = commitment, mô phỏng nb03:
load best_params -> override config -> run_multi_seed. Test: max_length 256 + augment +
test 314-sạch + multi-seed mean±std @ NGƯỠNG 0.5 (không tự-chọn ngưỡng) + early-stop.
Temp output_dir + 2 seed -> KHÔNG đụng model thật. Chạy: python outputs/_smoke_train_phase03.py
"""
import json
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.training.train_model import (load_yaml_config, resolve_runtime_config,
                                       run_multi_seed)

TMP = Path("outputs/_smoke_commitment")
RAW = load_yaml_config(Path("config/train.yml"))
cfg = resolve_runtime_config(RAW, task="commitment")

bp_path = Path("outputs/models/commitment/best_params_commitment.json")
if bp_path.exists():
    bp = json.load(open(bp_path))
    if "max_length" in bp:
        cfg["model"]["max_length"] = bp.pop("max_length")
    cfg["training"].update(bp)
    print(f"[smoke-train] best_params: {bp}")

cfg["paths"]["output_dir"] = str(TMP)
SEEDS = [42, 43]
print(f"[smoke-train] max_length={cfg['model']['max_length']} (kỳ vọng 256) "
      f"| seeds={SEEDS} | epochs={cfg['training']['epochs']} (early-stop)")

try:
    m = run_multi_seed(cfg, SEEDS)
    print("\n[smoke-train] keys:", list(m.keys()))
    print("[smoke-train] test @0.5:", m.get("test"))
    print("[smoke-train] inference_threshold:", m["inference_threshold"])

    # --- verify CÁCH C ---
    assert m["model_saved_seed"] == 42
    assert len(m["per_seed"]) == 2
    assert m["inference_threshold"] == 0.5, "ngưỡng phải = 0.5 (cách C)"
    assert "threshold" not in m and "test_argmax" not in m, "vẫn còn field tự-chọn-ngưỡng cũ!"
    assert "threshold" not in m["per_seed"][0], "per_seed vẫn có threshold!"
    assert m["test"]["f1_positive"]["std"] is not None, "thiếu std"
    tp = pd.read_parquet(TMP / "test_predictions.parquet")
    assert len(tp) == 314, f"test size {len(tp)} != 314"
    assert (tp["sentence"].fillna("").str.strip() == "").sum() == 0, "còn câu rỗng!"
    saved = json.load(open(TMP / "metrics_summary.json", encoding="utf-8"))
    assert saved["inference_threshold"] == 0.5

    def g(test, k, s="mean"):
        v = test.get(k, {})
        return round(v.get(s, float("nan")), 4) if isinstance(v, dict) else v
    print("[smoke-train] nb03 row:",
          {"macro_f1": g(m["test"], "macro_f1"), "f1_pos": g(m["test"], "f1_positive")})
    print("\n[smoke-train] PHASE 03 TRAIN SMOKE PASS (cách C, ngưỡng 0.5) ✅")
finally:
    shutil.rmtree(TMP, ignore_errors=True)
    print("[smoke-train] dọn", TMP)
