# Spec 02 — Mô hình phân loại

Backbone chung: `vinai/phobert-base` (max_length 256, **bắt buộc word-segment**
bằng VnCoreNLP/pyvi trước khi tokenize — PhoBERT train trên text đã tách từ).
Baseline so sánh: zero-shot `xlm-roberta-base` (train EN, infer VI — không cần dịch).

## 1. M1 — Topic E/S/G (multi-label, partial labels)

- Kiến trúc: PhoBERT + 3 đầu sigmoid (env, soc, gov).
- Loss: **masked BCE** — chỉ tính loss trên cột có nhãn (spec 01 §3).
- Split: stratified 80/10/10 trên từng tập nguồn trước khi gộp (tránh leak;
  giữ tỉ lệ positive per pillar).
- Ngưỡng quyết định: tune per-pillar trên dev (maximize F1), không mặc định 0.5.
- Eval: (a) per-pillar F1 trên dev/test dịch; (b) Macro-F1 trên VN human-eval set;
  (c) upper-bound: cùng kiến trúc train+test trên EN gốc.
- Câu không trụ nào vượt ngưỡng → `non_esg`, loại khỏi các tầng sau.

## 2. M2 — Claim model: Commitment + Specificity (multi-task, 2 đầu)

- Vì 2 task chia sẻ đúng cùng tập văn bản (đã xác minh), train **một** PhoBERT
  2 đầu sigmoid: `is_commitment`, `is_specific`. Loss = BCE(commitment) + BCE(specificity).
- Ưu điểm: encoder học biểu diễn nhất quán; inference 1 lượt; ít tham số hơn.
- Fallback (nếu multi-task kém hơn trên dev): 2 mô hình single-task riêng — giữ cả 2 trong code,
  chọn theo dev, báo cáo cả 2 trong ablation.
- **Augment cho khoảng trống S/G (L2):** thêm `action_500` (ESG-wide) vào đầu commitment
  (action=1 → commitment-side positive theo định nghĩa ClimateBERT "commitment/action");
  **ML-Promise** EN+FR+JA 1.200 mẫu dịch VI (đã có): nhãn promise yes/no (full ESG)
  augment thêm đầu commitment.
  ~~Auxiliary head thứ 3 từ env_claims~~ **ĐÃ BỎ** (quyết định 2026-06-12): không dùng
  output, lợi ích regularization chưa chứng minh, và env_claims climate-only có nguy cơ
  kéo encoder lệch thêm về E (ngược mục tiêu vá S/G); env_claims chỉ còn dùng cho
  augment topic-E (spec 01 §3).
- Phạm vi áp dụng khi inference: chỉ chạy trên câu đã qua M1 (topic ≠ non_esg).
- Eval: F1 per head trên test dịch + VN human-eval set, **tách theo trụ E vs S/G**
  để định lượng domain shift (climate→S/G) thay vì chỉ thừa nhận suông.

## 3. Quy trình train chung

- Tokenizer use_fast=False (PhoBERT), seed cố định, 5 seeds cho kết quả chính (mean±std).
- Hyperparams khởi điểm: lr 2e-5, batch 32, epochs 5, warmup 10%, weight decay 0.01;
  tune nhẹ trên dev (lr, epochs) — không grid search lớn.
- Class imbalance: pos_weight trong BCE theo tỉ lệ nhãn.
- Lưu: `outputs/models/<task>/`, metrics JSON `outputs/metrics/<task>.json`,
  config snapshot kèm theo (tái lập).
- Baseline tối thiểu phải vượt: TF-IDF + Logistic Regression per task.

## 4. Ma trận thí nghiệm transfer (T6)

| Trục | Giá trị |
|---|---|
| Phương pháp | zero-shot XLM-R / translate-train PhoBERT |
| Lọc nhiễu | none / QE / QE+CL |
| Task | topic, commitment, specificity |

Kết luận chọn cấu hình tốt nhất theo VN human-eval set; báo cáo cả kết quả null nếu có.
