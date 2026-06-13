# Đánh giá CTI + framing (brainstorm 2026-06-13)

> Đánh giá lại tính đúng đắn của chỉ số CTI (xây dựng + sử dụng) trước khi chạy Phase 5.
> Tham chiếu: Bingler et al. 2022 (*Cheap talk and cherry-picking*, Finance Research Letters);
> code `src/pipeline/cti.py` + `classify_corpus.py`.

## 1. Định nghĩa CTI — ✅ ĐÚNG
CTI = #{cam kết mơ hồ (specificity=0)} / #{cam kết} = trung thành Bingler 2022 ("share of precise vs
imprecise climate commitments"). Chỉ là tỉ lệ output 2 classifier đã công bố, không trọng số tay
→ luận điểm phòng thủ tốt (khác EWRI).

## 2. Xây dựng — đúng công thức, 4 rủi ro
- **Cascade 3 tầng** (topic→commitment[chỉ ESG]→specificity[chỉ commit]) → lỗi CỘNG DỒN.
- **Bias có hướng → CTI lệch CAO:** specificity recall(cụ thể)=0.67 → 33% câu cụ thể bị gán "mơ hồ"
  (vs 7% chiều ngược) → tử số phình → **CTI tuyệt đối thổi phồng washing**.
- **commitment precision 0.64** → ~36% "cam kết" giả → mẫu số phình (thêm nhiễu).
- **Domain mismatch:** commitment+specificity train climate, áp cả S/G → CTI cho S/G kém tin hơn E.
- **min_commit=5 quá thấp** + granularity per-trụ mịn (Bingler đo firm-year/14k báo cáo) → mẫu nhỏ, nhiễu.

## 3. Sử dụng — 4 điểm bắt buộc
- **CTI tuyệt đối KHÔNG tin → chỉ SO SÁNH TƯƠNG ĐỐI (xếp hạng bank)** + bootstrap CI.
- **Proxy nội-văn-bản, KHÔNG đo "walk"** (hành vi thật) — VN không có walk-data ngoài.
- **Đọc cùng selective disclosure + min-n** (bank né trụ → CTI trụ đó vô nghĩa).
- 🔴 **THIẾU construct validation** trong plan → CTI chưa chứng minh "đo washing".

## 4. Quyết định (brainstorm)
- **Grounded-CTI: CTI thuần TRƯỚC (Phase 5), grounded-CTI SAU (mở rộng); báo cáo CẢ 2.** CTI thuần =
  nền Bingler-faithful; grounded-CTI (thêm NLI tìm bằng chứng) = đóng góp MỚI; chênh lệch = "cheap talk ẩn".
- **Framing CTI (đề xuất, chốt):**
  1. Đơn vị chính = **pooled (bank × trụ, gộp năm)** + `min_commit`~30; per-year phụ lục.
  2. Diễn giải **tương đối** (ranking) + CI, KHÔNG tuyệt đối.
  3. Tách **E vs S/G** (S/G kém tin — domain).
  4. Đọc cùng selective disclosure + min-n.
  5. Framing **proxy nội-văn-bản**, khai báo bias-lệch-cao.
  6. Sensitivity quét ngưỡng → Kendall τ (đã có).
- **Validation: PENDING — user bàn với đồng nghiệp.** Talking-points:
  - Synthetic (code-được, rẻ): trộn câu mơ hồ → CTI tăng đơn điệu. **Tối thiểu nên có.**
  - Known-group (cần gán tay `bank_signals.csv` 10×5×~5 cột): nhóm có-tín-hiệu → CTI thấp hơn (Mann-Whitney). Mạnh nhất.
  - Khuyến nghị: synthetic bắt buộc; known-group nếu có người gán.

## 5. Việc cần làm (cập nhật Phase 5 + thêm phase mới)
- **Phase 05 (CTI thuần):** sửa `cti.py` theo framing (pooled+min30, tách E/S/G) + chạy trên model gold + sensitivity. Sửa note "silver/circular" → gold.
- **Phase 06 (MỚI — Index validation):** synthetic (bắt buộc) + known-group (nếu chốt) → đây là điều kiện để CTI có nghĩa khoa học. **Hiện plan 20260613 CHƯA có phase này — cần thêm.**
- **Phase 07 (MỞ RỘNG — grounded-CTI):** lớp grounding (retrieval bkai + NLI mDeBERTa + evidence pool + θ sweep) → grounded-CTI. Sau khi CTI thuần + validation xong.

## 6. Câu hỏi chưa giải quyết
1. Validation: cả 2 hay chỉ synthetic? Ai gán `bank_signals.csv`? (bàn đồng nghiệp)
2. grounded-CTI: NLI cross-lingual VN (`mDeBERTa-xnli`) có đủ tốt cho tiếng Việt không? (rủi ro phase 07)
3. Phase 04 (VN human-eval) vẫn là điều kiện cần TRƯỚC khi tin mọi số CTI (model train climate-dịch).
