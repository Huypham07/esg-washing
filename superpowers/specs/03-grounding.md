# Spec 03 — Grounding: retrieval bằng chứng + NLI

Mục tiêu: với mỗi câu **commitment**, tìm trong **cùng báo cáo** (cùng doc_id)
các span bằng chứng và đo mức độ được "đỡ" (evidence support), tạo tín hiệu cho grounded-CTI.

## 1. Định nghĩa candidate evidence pool (quan trọng — thu hẹp, không lấy cả corpus)

Một câu là **ứng viên bằng chứng** nếu thuộc cùng doc_id với claim VÀ thỏa ít nhất một:
- chứa số liệu (regex: số + đơn vị %, tỷ, triệu, tấn CO2, MWh, kWh, ha, tỷ đồng…), hoặc
- `block_type` ∈ {table, list} (số liệu dạng bảng), hoặc
- được M2 dự đoán `is_specific = 1` và `is_commitment = 0` (fact cụ thể, không phải lời hứa).

Loại câu chính nó và các câu trùng block với claim (tránh tự đỡ chính mình).

## 2. Retrieval

- Bi-encoder: `bkai-foundation-models/vietnamese-bi-encoder`.
- Encode claim + pool; lấy top-k (k=5) theo cosine, ngưỡng sàn sim ≥ 0.5
  (dưới sàn coi như không có ứng viên).
- Index per-document (pool nhỏ, vài nghìn câu/doc) — không cần FAISS toàn corpus.

## 3. NLI claim–evidence

- Model: `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` (XNLI có tiếng Việt — đã kiểm tra).
- Input: premise = evidence (kèm ctx_prev/ctx_next nếu ngắn), hypothesis = claim.
- **Giữ nguyên phân phối** {entail, neutral, contradict} — KHÔNG argmax nhị phân.

## 4. Evidence-support score (xử lý hạn chế L5)

Entailment chặt hiếm khi đúng cho "cam kết tương lai vs dữ kiện quá khứ". Định nghĩa:

```
support(claim) = max_{e ∈ top-k} P_entail(e, claim)
```

- **Truy vết related work:** kiến trúc retrieval top-k + NLI per-pair rồi aggregate là
  pipeline chuẩn của FEVER (Thorne 2018) và CLIMATE-FEVER (Diggelmann 2020); max-aggregation
  trên các evidence là cách aggregate được dùng trong các hệ verification câu-mức
  (e.g. Schuster 2021 — VitaminC; Stammbach & Ash 2020). Không thêm bất kỳ trọng số tự chế nào.
- Một claim là **grounded** nếu `support ≥ θ`. θ là tham số tự do duy nhất →
  không chọn cứng: quét θ ∈ {0.5, 0.7, 0.9} trong sensitivity analysis (spec 05),
  báo cáo grounded-CTI theo cả 3 mức; bảng chính dùng 0.7 và phải nói rõ đây là lựa chọn trình bày.

## 5. Human spot-check

- Lấy 100 cặp (claim, top-1 evidence) phân tầng theo bucket support → người đọc gán
  {đỡ thật / liên quan nhưng không đỡ / không liên quan}.
- Báo cáo precision của cờ grounded tại từng θ. Đây là evidence chính cho tính hợp lệ của lớp này.
- Khi có **ML-Promise** (spec 01 §7.1): dùng nhãn *supporting evidence yes/no* +
  *clarity* (dịch sang VI) làm gold ngoài để đánh giá evidence-support score —
  nguồn validation độc lập duy nhất cho lớp grounding.

## 6. Output

`outputs/grounding/claims_grounded.parquet`:
`sent_id, doc_id, bank, year, pillar, is_specific, top_evidence_ids, support, grounded@{0.5,0.7,0.9}`
