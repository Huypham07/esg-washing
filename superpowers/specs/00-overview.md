# Spec 00 — Tổng quan & cơ sở lý thuyết

> Đo lường ESG-washing (talk-vs-walk decoupling) trong báo cáo thường niên/bền vững
> của ngân hàng thương mại Việt Nam, bằng pipeline NLP cross-lingual (translate-train EN→VI).

## 1. Định vị bài toán

- **Hiện tượng:** báo cáo dày đặc cam kết ESG nhưng mơ hồ, thiếu số liệu, thiếu bằng chứng.
- **Khung lý thuyết:**
  - *Decoupling* giữa policy-talk và practice-walk — Meyer & Rowan (1977); Bromley & Powell (2012).
  - *Cheap talk* — Crawford & Sobel (1982): phát ngôn không tốn chi phí, không ràng buộc → tín hiệu yếu.
  - *Cheap Talk Index* — Bingler, Kraus, Leippold, Webersinke (2022, Finance Research Letters):
    tỉ lệ cam kết khí hậu **không cụ thể** trên tổng cam kết. Đây là chỉ số gốc mà nghiên cứu kế thừa.
  - *Claim verification* — FEVER (Thorne 2018), CLIMATE-FEVER (Diggelmann 2020): khung
    claim–evidence–NLI cho lớp grounding.
  - *Cross-lingual transfer* — translate-train là baseline chuẩn của XNLI (Conneau 2018) /
    XTREME (Hu 2020); hợp lệ cho tiếng Việt vì VI nằm trong XNLI.

## 2. Đóng góp dự kiến

1. **Grounded-CTI** — mở rộng CTI bằng lớp grounding nội văn bản (retrieval + NLI):
   một cam kết "specific" vẫn là cheap talk nếu không có span bằng chứng đỡ trong cùng báo cáo.
2. **Full E/S/G cho ngân hàng VN** — nghiên cứu washing đầu tiên phủ cả 3 trụ cột,
   tiếng Việt, ngành ngân hàng (10 NHTM, 2020–2024, ~120k câu).
3. **Bộ dữ liệu translate-train VI công khai** — 6 task gold EN đã dịch máy sang VI,
   kèm quy trình lọc nhiễu (QE + Confident Learning).

## 3. Nguyên tắc thiết kế (chốt sau review đề cương)

- **Walk = proxy nội văn bản** (specificity + grounding). KHÔNG tuyên bố đo "hành vi thật" —
  framing trong paper là *intra-report substantiation*, vì VN không có walk-data ngoài
  (CDP/MSCI/RepRisk không phủ).
- **Không tự bịa bất kỳ công thức/trọng số/thang đo nào.** Mọi thành phần phương pháp
  (chỉ số, cách gộp, ngưỡng) phải truy vết được về related work đã công bố; tham số tự do
  duy nhất được phép là ngưỡng quyết định, và phải xử lý bằng sensitivity analysis
  thay vì chọn cứng. Chỉ số = tỉ lệ output của các classifier đã công bố.
- **Không dùng nhãn LLM cũ** (`data/legacy/labels/*/llm_prelabels.parquet`) làm nhãn train;
  có thể dùng làm weak-check phân phối.
- Corpus đầu vào: **chỉ build lại từ raw extract** (`data/extracted/raw_ocr_annual_report.zip`),
  không dùng các parquet dẫn xuất cũ trong `data/legacy/corpus/`.
- Mọi kết luận phải kèm uncertainty (bootstrap CI) và ngưỡng n tối thiểu.

## 4. Kiến trúc pipeline (4 tầng)

```
raw txt ─► [P1 Corpus] câu sạch + metadata (bank, year, section, block)
        ─► [P2 Classify] topic E/S/G (multi-label) → commitment → specificity
        ─► [P3 Ground]  retrieval evidence + NLI claim–evidence (giữ phân phối)
        ─► [P4 Index]   CTI / grounded-CTI / selective disclosure + bootstrap CI
```

Chi tiết từng tầng: spec 01 (data), 02 (models), 03 (grounding), 04 (indices), 05 (validation).

## 5. Hạn chế phải khai báo trong paper

| # | Hạn chế | Giảm nhẹ |
|---|---|---|
| L1 | Specificity gốc có inter-annotator agreement thấp (α≈0.17) → trần chất lượng nhãn | khai báo như trần chất lượng; human spot-check trên VN eval set |
| L2 | Commitment/Specificity train trên dữ liệu **climate** nhưng áp cho cả S/G | augment bằng `action_500` (ESG-wide); đánh giá riêng trên VN human-eval set theo trụ |
| L3 | Dịch máy gây nhiễu nhãn | QE filter + Confident Learning (spec 01 §4); ablation có/không lọc |
| L4 | Không có ground-truth washing | known-group + synthetic manipulation validation (spec 05) |
| L5 | NLI entailment chặt hiếm khi đúng cho cam kết tương lai vs bằng chứng quá khứ | dùng **phân phối** NLI làm evidence-support score, không nhị phân hoá (spec 03) |
