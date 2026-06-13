# Kết quả chính — Train 5 classifier (Phase 03)

> PhoBERT-v2, translate-train (gold dịch EN→VI), **5 seed [42–46] mean±std**, ngưỡng quyết định **0.5**
> (đã cân lệch lớp bằng class-weights). Test = vi_gold dịch (miền climate/doanh nghiệp chung).
> Reproduce: `python outputs/_eval_phase03_results.py`.

## Bảng kết quả (sắp theo độ mạnh)

| Task | macro-F1 (mean±std) | F1 lớp dương | sàn TF-IDF | hơn sàn | Hạng |
|---|---|---|---|---|---|
| **env** (Môi trường) | **0.954** ± 0.016 | 0.939 | 0.865 | +0.09 | Mạnh |
| **soc** (Xã hội) | **0.911** ± 0.012 | 0.891 | 0.844 | +0.07 | Mạnh |
| **specificity** (tính cụ thể) | **0.808** ± 0.011 | 0.748 | 0.735 | +0.07 | Khá |
| **gov** (Quản trị) | **0.801** ± 0.017 | 0.717 | 0.759 | +0.04 | Yếu — khó |
| **commitment** (cam kết) | **0.789** ± 0.008 | 0.729 | 0.739 | +0.05 | Khá — khó |

## Đặc tính lỗi (lớp dương, seed 42) — quan trọng cho CTI
| Task | Precision | Recall | Ý nghĩa cho CTI |
|---|---|---|---|
| commitment | 0.64 | **0.84** | bắt gần đủ cam kết (mẫu số đầy đủ), lẫn cam kết "giả" → mẫu số hơi phình |
| specificity | **0.85** | 0.67 | sót ~1/3 câu cụ thể → đếm DƯ "mơ hồ" → **CTI lệch CAO hệ thống** |
| gov | 0.72 | 0.79 | — |

## 6 kết luận chính
1. **Phân tầng rõ:** Topic E/S **mạnh** (0.91–0.95); G + claim (commitment/specificity) **trung bình** (0.79–0.81).
2. **Cả 5 vượt sàn TF-IDF** → mô hình có giá trị thật. Nhưng **gov (+0.04) và commitment (+0.05)** chỉ nhỉnh hơn sàn ít → đây là 2 task **khó nhất** (kể cả PhoBERT cũng chật vật).
3. **Ổn định cao:** std ≤ 0.02, **không seed nào sụp** → số đáng tin (giá trị của multi-seed: báo cáo "0.80±0.02" thay vì 1 con số may rủi).
4. **commitment** recall cao (0.84) → tốt cho độ phủ mẫu số CTI; **specificity** recall thấp (0.67) → **đẩy CTI lệch cao** (phải xử lý bằng sensitivity Phase 05 + khai báo).
5. **Augment action_500 cho commitment: lợi không đáng kể** (macro gần như không đổi) — ghi nhận trung thực.
6. **Caveat lớn nhất:** đây là điểm trên **bản DỊCH (climate)**, CHƯA chứng minh chạy đúng trên **báo cáo bank VN thật** → **VN human-eval (Phase 04) là điều kiện cần** trước khi tin số CTI.

## Hạn chế phải khai báo (luận văn)
- **gov khó** — chỉ hơn sàn ~0.04, hết nguồn data gov → giới hạn claim về trụ Quản trị.
- **specificity:** nhãn gốc inter-annotator thấp (α≈0.17) + recall 0.67 → CTI lệch lên.
- **translate-train chưa verify domain bank VN** → phụ thuộc human-eval.

## Câu hỏi mở
- commitment có nên thử augment ml_promise (đang defer) để tăng cam kết S/G không? (lợi chưa chắc)
- Ngưỡng 0.5 cho specificity có làm CTI lệch nhiều không → đo ở Phase 05 sensitivity.
