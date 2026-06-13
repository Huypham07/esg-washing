# Phase 02 — Train/tuning hybrid upgrade (code, chưa train)

**Context:** recommendation D3 · tham chiếu `git show origin/optimize-research:src/esgwash/models/{trainer,tuning}.py`
**Priority:** P0 · **Status:** ✅ done (2026-06-13, code-reviewed) · **Depends:** 00
**Chốt:** warmup-only (KHÔNG dropout) · ~~threshold theo f1_positive~~ → **REVISED 2026-06-13: NGƯỠNG 0.5 cố định (cách C)** — bỏ tự-chọn ngưỡng (val nhỏ overfit → chọn 0.08 hại test + méo CTI; class-weights đã cân lệch; 0.5 nhất quán + dễ bảo vệ).

## Kết quả (✅ done)
Sửa `train_model.py` (threshold-tuning f1_positive + multi-seed mean±std + `inference_threshold`), `tune_hyperparams.py` (`OptunaPruningCallback` + warmup search, bỏ epochs), `config/train.yml` (seeds + epochs trần 10). **Smoke gov** (2ep/3trial/2seed, temp dir + in-memory study) PASS: pruner CẮT thật (intermediate=[3,1,1]), threshold-tuning lên f1_pos (0.727 vs argmax 0.715). **Code-review 3 fix đã xử lý:** [HIGH] `outputs/_run_train_eval_phobert.py` (runner Phase 03) cho schema mean±std mới; [LOW] thêm `inference_threshold`=seed[0] (Phase 05 đọc cái này); [LOW] tie-break ngưỡng `>=`. **Re-check:** bỏ `trainer.evaluate()` thừa (step pruning ảo). KHÔNG đụng build_model/decoder_lora.

**🔴 REVISED (Phase 03 train test, 2026-06-13):** chạy thử train thật lộ ra threshold-tuning f1_positive chọn ngưỡng **0.08** trên val nhỏ commitment (149) → **hại test** (f1_pos 0.730 < argmax 0.734) + sẽ **méo CTI** (gán ~mọi câu = commitment → phình mẫu số). → **Chuyển CÁCH C: ngưỡng 0.5 cố định**, bỏ `tune_threshold`. Giữ `_prob_positive`+`evaluate_split(threshold=)` cho Phase 05 sensitivity. metrics_summary: `inference_threshold=0.5`, bỏ `test_argmax`/`threshold`.

## Overview
Nâng `train_model.py` + `tune_hyperparams.py` + `config/train.yml`: threshold-tuning per-model, multi-seed mean±std, search space mới (bỏ epochs, thêm **warmup** — KHÔNG dropout), sửa pruner report-intermediate. **GIỮ** HF Trainer + WeightedTrainer + EarlyStopping. **KHÔNG** đụng kiến trúc/loss. **KHÔNG** touch `build_model.py` (vì bỏ dropout).

## Key insights / quyết định
- **Threshold theo f1_positive** (không phải macro_f1): CTI dựng HOÀN TOÀN từ dự đoán lớp positive (commitment=1, specificity=0); macro_f1 pha loãng bằng F1 lớp-âm-dễ. **Báo cáo CẢ macro_f1 + f1_positive** (code đã tính sẵn cả 2). Tune trên VAL, áp TEST.
  - Cảnh báo: tối đa f1_positive có thể đẩy ngưỡng thấp → over-predict → phình mẫu số CTI → **Phase 05 sensitivity (Kendall τ) là lưới an toàn bắt buộc**.
- **Multi-seed 5** (Dodge2020/Mosbach2021: BERT data nhỏ bất ổn theo seed) → mean±std.
- **Warmup-only (KHÔNG dropout):** warmup là lever ổn định đúng theo lit (Mosbach/Dodge: warmup+lr thấp, KHÔNG phải dropout). Trên val nhỏ (gov ~198) ít chiều HPO = ít overfit-val; lại đỡ touch build_model. Dropout lợi mơ hồ → bỏ.
- **Pruner** hiện vô dụng (objective chỉ trả best cuối, không report intermediate) → thêm callback report **dev macro_f1** (= metric model-selection) mỗi epoch → MedianPruner prune thật.
- **threshold lưu `metrics_summary.json`** (KHÔNG ghi vào HF `config.json` của model để tránh clobber) → Phase 05 CTI đọc.

## Requirements
- threshold per-model (tune f1_positive trên val) áp inference + ghi metrics_summary.json.
- multi-seed [42-46] → test mean±std; tune vẫn 1 seed cố định (tiết kiệm).
- search: lr·batch·wd·**warmup** (bỏ epochs + KHÔNG dropout).

## Related Code Files
- **Modify:** `src/training/train_model.py` —
  (a) `predict_proba` (softmax → P(class1)); `tune_threshold(prob_val, y_val)` quét 0.05–0.95 **max f1_positive**; `evaluate_split` nhận `thr` → `(P≥thr)` thay argmax; lưu `threshold` + cả macro_f1/f1_positive vào `metrics_summary.json`.
  (b) `run_multi_seed(config)` gọi `train_once`+tune_threshold+eval mỗi seed, gom mean±std (threshold tuned per-seed), lưu model seed[0]. `main()` đọc `training.seeds`.
- **Modify:** `src/training/tune_hyperparams.py` — search mới (bỏ `epochs`, thêm `warmup_ratio`[0,0.2]; **KHÔNG dropout** → KHÔNG touch build_model); `OptunaPruningCallback(TrainerCallback)` gọi `trial.report(dev_macro_f1, epoch)`+`should_prune()→TrialPruned`; `train_once(config, trial=None)` thêm callback khi có trial; giữ TPE+SQLite resumable.
- **Modify:** `config/train.yml` — anchor `_tune_*` thêm `warmup_ratio` range (bỏ dropout, bỏ epochs khỏi search); `seeds: [42,43,44,45,46]` vào `_train`; `epochs` nâng trần ~10 + early-stopping.
- **KHÔNG đụng:** `build_model.py` (không dropout), nhánh `decoder_lora` (giữ bake-off LLM).

## Implementation Steps
1. train_model: `predict_proba`; `tune_threshold` (val, max **f1_positive**) → `best_thr`; eval test dùng best_thr; ghi threshold + macro_f1 + f1_positive vào `metrics_summary.json`.
2. train_model: `run_multi_seed` — loop seeds, `train_once` (save seed[0]), test metrics per seed (threshold tuned per seed), `aggregate{mean,std}` cho cả macro_f1 + f1_positive.
3. tune_hyperparams: `OptunaPruningCallback`; `train_once(trial=...)` gắn callback; `objective` bỏ epochs, thêm warmup (giữ lr/batch/wd); epochs cố định = max.
4. config: cập nhật anchor; `resolve_runtime_config` truyền seeds + epochs đúng.
5. Smoke: `--task gov --trials 2` + `run_multi_seed` 2 seed gov — verify không lỗi, có trial pruned, threshold + mean±std xuất ra.

## Todo
- [ ] `tune_threshold` (max f1_positive) + lưu/áp threshold (metrics_summary.json)
- [ ] `run_multi_seed` mean±std (cả macro_f1 + f1_positive)
- [ ] `OptunaPruningCallback` report-intermediate (dev macro_f1)
- [ ] Search space: bỏ epochs, thêm warmup, seeds (KHÔNG dropout) + config anchor
- [ ] Smoke gov 2-trial + 2-seed OK; compile sạch; decoder_lora không vỡ

## Success Criteria
- `tune_hyperparams --task gov --trials 2` chạy, log có trial bị prune (khi đáng).
- `metrics_summary.json` có `threshold`, `macro_f1`, `f1_positive` (mean/std qua 5 seed).
- Threshold đọc lại được ở inference (Phase 05 dùng); decoder_lora vẫn chạy.

## Risks
- HF Trainer callback + Optuna: callback đọc đúng key (`eval_macro_f1`) trong `on_evaluate`.
- threshold per-seed khác nhau → model lưu (seed[0]) đi kèm threshold riêng; mean±std là của metric.
- threshold-tuning f1_positive có thể đẩy ngưỡng thấp → KIỂM bằng Phase 05 sensitivity trước khi diễn giải CTI.
- Đừng để threshold-tuning chạm test (chỉ val).
