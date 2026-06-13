# Phase 03 — Re-tune + re-train 5 models

**Priority:** P0 · **Status:** pending · **Depends:** 01 (data), 02 (code)

## Overview
Chạy pipeline đã nâng cấp trên 5 model: re-tune (1 seed, search mới) → train cuối 5-seed mean±std + threshold. So với baseline gold. Compute rộng rãi → làm đầy đủ.

## Key insights
- Đích chính: commitment lên (action_500 augment + threshold + multi-seed); gov lên (threshold + multi-seed — cross-label đã bỏ). specificity dự kiến ~phẳng (không augment) — chấp nhận + khai báo.
- Baseline topic giữ nguyên (không cross-label) → so trực tiếp env.955/soc.916/gov.821; cải thiện = thuần threshold+multi-seed.
- Tune 1 seed (rẻ), final 5 seed (rigor). KHÔNG tune lại mỗi seed.

## Requirements
- 5 model re-tune với search mới (~30 trial/model, resumable SQLite).
- Final train 5-seed → `metrics_summary.json` mean±std + threshold + so baseline.

## Related Code Files
- Modify/dùng: `src/training/train_model.py`, `tune_hyperparams.py`, `config/train.yml` (đã sửa Phase 02). Output: `outputs/models/<task>/`, `outputs/metrics/`.
- Refresh: `src/training/eval/baseline_classifiers.py` + `compare_models.py` → bảng so sánh mới.

## ✅ KẾT QUẢ (2026-06-13) — bản tổng hợp: `results-train-classifiers.md`
5 classifier (5-seed mean±std @ ngưỡng 0.5): **env 0.954±0.016 · soc 0.911±0.012 · specificity 0.808±0.011 · gov 0.801±0.017 · commitment 0.789±0.008** (macro-F1). Tất cả vượt sàn TF-IDF, std ≤0.02 (ổn định, không seed sụp). gov+commitment khó nhất (chỉ +0.04-0.05 trên sàn). commitment recall 0.84/prec 0.64; specificity prec 0.85/recall 0.67 (→ CTI lệch cao, xử lý ở Phase 05). Caveat: test=bản dịch climate, chưa verify bank VN thật → cần Phase 04. Reproduce: `python outputs/_eval_phase03_results.py`.

## Setup (2026-06-13) — USER tự chạy qua notebook
- Kết quả cũ **đã archive** (recoverable): `outputs/_archive_pre_phase03/` (5 study `.db`, `outputs/models/*`, `outputs/compare/*`). Workspace sạch.
- Notebook đã cập nhật cho setup mới (multi-seed + threshold + warmup-search + study mới).

## Cách chạy (USER)
1. **Tune:** mở `notebooks/02-tune-classifiers.ipynb` → run all (5 task × `N_TRIALS=30`, ~2.5-5h). Ra `best_params_<task>.json`.
2. **Train:** mở `notebooks/03-train-classifiers.ipynb` → run all (5 task × 5 seed, ~1-2h). Ra model + `metrics_summary.json` (mean±std + `inference_threshold`) + bảng kết quả + baseline.
3. **Phân tích (cùng tôi sau khi chạy xong):** so bảng mới vs baseline (archived `compare/` + bảng baseline trong plan.md). `metrics_summary` có cả `test`(threshold) lẫn `test_argmax` → so **argmax-mới vs argmax-cũ** (cải thiện model thuần) + threshold-mới (bản triển khai). gov/commitment cải thiện? specificity flat (kỳ vọng)? Ghi nhận xét + limitation.

## Todo
- [ ] Re-tune 5 task (30 trial mỗi) — log số trial pruned
- [ ] Train 5-seed 5 task → mean±std + threshold
- [ ] Bảng so sánh mới vs baseline
- [ ] Nhận xét gov/commit/spec

## Success Criteria
- 5 model có `metrics_summary.json` với mean±std (5 seed) + threshold.
- gov & commitment macro-F1 ≥ baseline gold (.821/.795) — lý tưởng cải thiện rõ; nếu không, có giải thích.
- Bảng compare mới đủ cho paper (per-task, mean±std, vs tfidf_lr floor).

## Risks
- gov không cải thiện đủ bằng threshold+multi-seed → chấp nhận ~.82 (hết nguồn data gov; cross-label đã đo ra marginal). commit chưa đủ → cân nhắc ml_promise (deferred) vòng sau.
- specificity flat = kỳ vọng (không augment) → KHÔNG cố ép; khai báo limitation.
- Variance 5-seed lớn ở data nhỏ (gov/commit/spec) → đó chính là lý do báo cáo std (trung thực).
