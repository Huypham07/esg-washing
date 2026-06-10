# Spec 05 — Validation chỉ số (không có ground-truth washing)

Vì không có nhãn washing thật, tính hợp lệ của CTI/grounded-CTI được lập luận bằng
**construct validation** — chuỗi kiểm định chuẩn trong văn liệu đo lường
(known-group validity: Cronbach & Meehl 1955; áp dụng cho chỉ số văn bản: Grimmer & Stewart 2013).
Không phép kiểm định nào dưới đây dùng công thức tự chế.

## V1 — Known-group validity

- Dữ liệu: `data/external/bank_signals.csv` (spec 01 §6 — assurance, GRI, VNSI,
  green credit, sáng kiến quốc tế; gán tay từ nguồn công khai).
- Giả thuyết: nhóm bank-year có tín hiệu thực chất (vd. có assurance độc lập)
  có CTI/grounded-CTI **thấp hơn** nhóm không.
- Kiểm định: Mann-Whitney U (phi tham số, mẫu nhỏ) per signal; báo cáo effect size
  (rank-biserial). Kết luận kể cả khi null.

## V2 — Synthetic manipulation (sanity check hướng phản ứng)

- Lấy báo cáo thật, tạo bản "washed hơn": thay câu specific bằng câu non-specific
  cùng trụ (rút từ chính corpus, do M2 gán), theo các mức trộn 10/30/50%.
- Kỳ vọng: CTI tăng đơn điệu theo mức trộn. Đây là perturbation test kiểu
  CheckList (Ribeiro 2020) cho chỉ số, không phải đánh giá độ chính xác tuyệt đối.
- Tương tự chiều ngược: bơm câu specific+grounded → grounded-CTI phải giảm.

## V3 — Sensitivity / robustness

- Quét θ grounding ∈ {0.5, 0.7, 0.9} (spec 03): ranking giữa các bank có ổn định không —
  đo bằng Kendall's τ giữa các bảng xếp hạng.
- Quét ngưỡng quyết định của M1/M2 (±0.05 quanh ngưỡng tune): CTI dao động bao nhiêu.
- Bootstrap CI (spec 04 §4) là lớp uncertainty mặc định cho mọi con số.

## V4 — Chất lượng từng tầng mô hình (điều kiện cần)

- M1/M2 phải đạt F1 chấp nhận được trên **VN human-eval set** (spec 01 §5) trước khi
  chỉ số được diễn giải; nếu transfer kém ở S/G thì giới hạn claim của paper về S/G.
- Grounding: precision human spot-check 100 cặp (spec 03 §5).

## Bảng thí nghiệm tổng (khớp T-list trong đề cương)

| Exp | Nội dung | Output |
|---|---|---|
| E1 | Baselines TF-IDF+LR vs PhoBERT per task | metrics/exp1.json |
| E2 | Upper-bound EN (train+test EN gốc) | metrics/exp2.json |
| E3 | zero-shot XLM-R vs translate-train (× lọc nhiễu) | metrics/exp3.json |
| E4 | Grounding: grounded share + human spot-check | metrics/exp4.json |
| E5 | Ablation QE / QE+CL | metrics/exp5.json |
| E6 | Known-group (V1) | metrics/exp6.json |
| E7 | Synthetic manipulation (V2) | metrics/exp7.json |
| E8 | Sensitivity (V3) | metrics/exp8.json |
