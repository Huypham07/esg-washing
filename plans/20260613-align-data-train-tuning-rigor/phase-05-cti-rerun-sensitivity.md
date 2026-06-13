# Phase 05 — CTI re-run + sensitivity analysis

**Context:** recommendation §3 coupling · downstream `src/pipeline/{classify_corpus,cti,run_cti,cti_figures}.py`
**Priority:** P1 · **Status:** pending · **Depends:** 03 (model mới)

## ✅ KẾT QUẢ (SƠ BỘ — 2026-06-13)
Classify 59,985 câu × 5 gold model → CTI pooled (27 ô, đều ≥30 cam kết, loại bsc). **CTI mean: E 0.77 · S 0.81 · G 0.94** (TƯƠNG ĐỐI — không tuyệt đối, đã thổi phồng do specificity bias + commitment precision). 🔴 **Thứ hạng trụ NHIỄU bởi chất lượng classifier:** E thấp nhất (specificity in-domain nhận ra "cụ thể"), **G .94 phần lớn ARTIFACT** (specificity climate không hiểu "cụ thể" quản trị). Sensitivity Kendall τ: **E→1.0 (đáng tin nhất), G .67 (kém tin nhất)**. Dùng được: ranking bank TRONG trụ E. Output: `outputs/index/{cti_pooled_gold,cti_gold,selective_disclosure_gold}.parquet` + `cti_summary_gold.json`. **SƠ BỘ — chờ Phase 04 (verify VN thật) + Phase 06 (validation).**

## Overview
Chạy lại downstream CTI trên model gold-rigor mới (thay silver) + THÊM sensitivity analysis (bắt buộc vì threshold-tuning ảnh hưởng trực tiếp CTI). Code CTI giữ nguyên, thêm lớp sensitivity.

## Key insights
- Số CTI demo cũ (E.52/S.54/G.60) chạy trên **silver** → sẽ ĐỔI khi dùng model gold (out-domain) → diễn giải lại.
- threshold-tuning commit/spec dịch chuyển CTI → phải chứng minh ranking ỔN ĐỊNH qua sensitivity (Kendall τ), nếu không = p-hacking chỉ số.
- 🔴 bsc (BIDV Securities, OCR mất dấu → CTI .76 ARTIFACT) — xử lý/loại, đừng trình "worst-washer".

## Requirements
- Classify corpus bằng 5 model mới + threshold tuned.
- CTI + bootstrap CI + selective disclosure tái chạy.
- Sensitivity: quét ngưỡng commit/spec ±0.05 → Kendall τ ranking bank; báo cáo 3 mức.

## Related Code Files
- **Modify:** `src/pipeline/classify_corpus.py` (load model+threshold mới), `src/pipeline/run_cti.py` (track=gold), `src/pipeline/cti.py` (nâng min_commit 5→ cân nhắc theo n; thêm hàm sensitivity).
- **Create (hoặc thêm vào cti.py):** `cti_sensitivity(enriched, thr_deltas, ...)` → Kendall τ giữa ranking ở các ngưỡng.
- Output: `outputs/index/cti_gold.parquet`, `outputs/metrics/cti_sensitivity.json`, figures cập nhật.

## Implementation Steps
1. `classify_corpus`: trỏ sang model gold mới, classify ở **ngưỡng 0.5** (cách C — `inference_threshold=0.5` trong metrics_summary; argmax). Xuất enriched mới.
2. `run_cti` track="gold": CTI per (bank,year,pillar) + bootstrap CI + selective disclosure. **Framing (chốt brainstorm 2026-06-13 — xem `cti-evaluation-and-framing.md`):** nâng `min_commit` 5→~30; **ranking POOLED (bank×trụ, gộp năm) là CHÍNH**, per-year phụ lục; **diễn giải TƯƠNG ĐỐI** (xếp hạng + CI, KHÔNG tuyệt đối — CTI lệch cao do cascade + specificity-recall bias); **tách E vs S/G** (S/G kém tin, domain mismatch); đọc cùng selective disclosure; framing **proxy nội-văn-bản, KHÔNG phải walk**. Đây là **CTI THUẦN** (grounded-CTI = Phase 07; validation = Phase 06).
3. **Sensitivity (đây là chỗ xử lý ngưỡng bài bản — cách C để dành việc này cho đây):** lặp ngưỡng commit/spec ∈ {0.4, 0.5, 0.6} (quanh 0.5) → tính CTI per cell → Kendall τ giữa các bảng ranking bank → ghi `cti_sensitivity.json`. Dùng `_prob_positive`+`evaluate_split(threshold=)`. (Sau, khi có grounding: thêm θ sweep.)
4. Xử lý bsc: fix OCR (NFC + khôi phục dấu nếu được) HOẶC loại khỏi ranking chính + ghi chú phụ lục.
5. Cập nhật `cti_figures.py` → hình mới (CTI gold + CI + sensitivity).

## Todo
- [ ] classify_corpus dùng model+threshold mới
- [ ] CTI gold + bootstrap + selective disclosure tái chạy
- [ ] `cti_sensitivity` (Kendall τ ngưỡng ±0.05)
- [ ] Xử lý/loại bsc artifact
- [ ] Figures cập nhật

## Success Criteria
- `cti_gold.parquet` + summary mới (diễn giải lại E/S/G vs demo cũ).
- `cti_sensitivity.json`: Kendall τ cho thấy ranking ổn định (hoặc nêu rõ chỗ không ổn).
- bsc không xuất hiện sai lệch trong bảng chính.

## Risks
- Model gold (out-domain) có thể classify corpus bank VN kém hơn silver → CTI nhiễu hơn; **phụ thuộc Phase 04** để biết model thật sự tốt trên VN không (nếu human-eval kém → CTI không đáng tin → quay lại cải thiện model trước khi diễn giải).
- min_commit thấp → cell n nhỏ; cân nhắc nâng + pooled (bank×pillar) cho ranking chính.
- Sensitivity lộ ranking KHÔNG ổn → trung thực báo cáo, có thể chỉ kết luận pooled.
