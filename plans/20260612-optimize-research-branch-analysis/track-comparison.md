# Bảng so sánh: hướng của tôi (dqd) vs đồng nghiệp (Huy)

> Đối chiếu code thực tế 2 track (2026-06-12). Track tôi = `src/training`+`src/pipeline` (local, uncommitted);
> track Huy = `src/esgwash` (origin/optimize-research). Cả hai cùng pivot grounded-CTI + translate-train.

## Triết lý
- **Tôi:** đa nhánh, demo-first, **xong downstream** (CTI→figures). Thử nhiều cách gán nhãn + LLM bake-off.
- **Huy:** một nhánh, rigorous-first, **sâu upstream** (multi-head + cross-label + validation). Downstream stub.

## 1. Nguồn nhãn & data
| Khía cạnh | Tôi | Huy |
|---|---|---|
| Translate-train (vi_gold ← en_gold/translate) | ✅ Nhánh B `prepare_gold_splits` | ✅ đường chính `gold_loader` |
| LLM silver in-domain (vi_silver) | ✅ gpt-4o-mini gán câu bank VN thật | ❌ bác bỏ (spec cấm nhãn LLM train) |
| Augment S/G (action_500, ml_promise) | ❌ đã bỏ 3 bộ phụ (2026-06-11) | ✅ thêm vào commitment |
| Cross-pillar topic | ❌ không (3 model rời, nhãn sạch riêng) | ✅ ESGBERT cross-label điền NaN |

## 2. Kiến trúc model
| Khía cạnh | Tôi | Huy |
|---|---|---|
| Topic | 3 model nhị phân RỜI | 1 model multi-label 3-sigmoid |
| Commitment+Specificity | 2 model RỜI | multi-task 1 encoder 2 đầu |
| Lớp model | AutoModelForSequenceClassification (softmax) | MultiHeadClassifier tự viết (sigmoid) |
| Loss | class-weighted CrossEntropy (sqrt_inverse) | masked BCE + pos_weight |
| Backbone | phobert-base-v2 | phobert-base |
| Số model | 5 (×2 với silver ≈ 10) | 2 |

## 3. Train & tuning
| Khía cạnh | Tôi | Huy |
|---|---|---|
| Engine | HF Trainer + WeightedTrainer | vòng train tự viết |
| Tuning | Optuna TPE+MedianPruner+SQLite | Optuna TPE+MedianPruner+SQLite |
| Search | lr·batch·wd·epochs | lr·batch·wd·warmup·dropout (epochs cố định) |
| Ngưỡng | argmax (0.5 ngầm) | tune per-head F1 trên dev |
| Seed | 1 (42) | 5 → mean±std |
| Pruning thực tế | pruner có nhưng không report intermediate | epoch_callback→report → prune thật |
| Bake-off LLM QLoRA (Qwen/Vistral) | ✅ có | ❌ không |

## 4. Eval & downstream
| Khía cạnh | Tôi | Huy |
|---|---|---|
| Test set | gold test + silver (vòng tròn) | gold test + VN human-eval 300 κ (plan) |
| CTI+bootstrap+selective disclosure | ✅ đã code + chạy corpus | ❌ stub |
| Grounding→grounded-CTI | file EWRI cũ | spec sẵn, stub |
| Validation (known-group/synthetic/sensitivity) | ❌ chưa | spec sẵn, stub |
| Code | src/training+src/pipeline organic | package src/esgwash + tests + specs |

## Tinh túy
- **Tôi đi trước:** downstream xong, 2 nguồn nhãn, LLM bake-off, tuning xong → pipeline chạy end-to-end.
- **Huy chặt hơn:** multi-head+masked-BCE, ESGBERT cross-label (chống confirmation-bias), augment S/G,
  threshold-tuning, 5-seed, validation + human-eval → bảo vệ được trước hội đồng.
- **Cùng nền:** translate-train, Optuna TPE+SQLite, PhoBERT.

## 3 khác biệt quyết định
1. Silver (tôi) vs translate-only+cross-label (Huy) → tôi CẦN VN human-eval để chứng minh silver không circular.
   Silver in-domain có lợi thế domain-match (bank VN thật) mà translate-train (climate EN dịch) không có →
   nếu có human-eval, "in-domain-silver vs out-domain-translate" thành đóng góp thật, không phải điểm yếu.
2. 5 model rời (tôi) vs 2 multi-head (Huy) → của tôi vẫn hợp lệ, chỉ kém elegant/efficient; KHÔNG bắt buộc đổi.
3. Downstream: tôi xong, Huy stub → lợi thế lớn nhất của tôi.

## Câu hỏi mở
- Test F1 env.94/soc.90 là trên gold (translate) hay silver test? (xác nhận để biết con số nào trình thầy).
- Có nên hợp nhất: giữ downstream của tôi + mượn cross-label/multi-head/validation của Huy?
