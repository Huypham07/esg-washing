# Evidence Pipeline: Luồng, tính năng và so sánh phiên bản

## 1) Luồng logic evidence trong pipeline

1. **Input**: DataFrame sau bước actionability (`actionability_sentences.parquet`).
2. **Rule detection** (`evidence_detector.process_dataframe`):
   - Gán `has_evidence`, `evidence_types`, `evidence_strength` (theo rule + similarity placeholder).
3. **Link claim-evidence** (`evidence_linker.run_linking_variant`):
   - Tìm câu bằng chứng theo từng variant.
   - Tính `similarity_score`, `best_evidence`, `num_evidence`, `search_method`.
   - Nếu bật NLI: thêm `nli_entailment_score`, `nli_label`.
4. **Evidence score cuối**:
   - Dùng `evidence_strength_v2` từ linker (kết hợp similarity + rule + NLI tùy variant).
5. **Đầu ra**:
   - File evidence-scored corpus để dùng cho EWRI + report.

---

## 2) Các phiên bản đang giữ lại

### A. `nli` (bản chính)

**Tính năng**

- Retrieval theo **window + document-level TF-IDF**.
- Chọn nhiều bằng chứng (`top_k_evidence=3`).
- Có bước kiểm tra **NLI bằng model**.

**Ưu điểm**

- Cân bằng tốt giữa chất lượng và tốc độ.
- Ít nhiễu hơn bản không NLI.
- Ổn định khi chạy thực tế dài.

**Nhược điểm**

- Chậm hơn `v1_window`.
- Tốn tài nguyên và thời gian hơn bản không NLI.

---

### B. `v1_window` (bản so sánh 1)

**Tính năng**

- Chỉ tìm bằng chứng trong cửa sổ cục bộ quanh câu claim.
- Không dùng document-level retrieval.
- Không dùng NLI.

**Ưu điểm**

- Nhanh nhất, dễ debug.
- Là baseline rõ ràng để so với bản chính.

**Nhược điểm**

- Dễ bỏ sót bằng chứng nằm xa claim.
- Dễ chọn nhầm câu gần nhưng không thực sự support.

---

### C. `no_nli` (bản so sánh 2)

**Tính năng**

- Giữ retrieval giống bản chính (`window + document-level`).
- Tắt NLI để đo đúng tác động của NLI.

**Ưu điểm**

- So sánh trực diện với `nli` để lượng hóa đóng góp của NLI.
- Nhanh hơn `v2_rule_nli`.

**Nhược điểm**

- Chất lượng xác nhận quan hệ claim-evidence thấp hơn khi câu gần nghĩa nhưng không entail.

---

## 3) Khi nào dùng phiên bản nào

- **Mặc định production**: `nli`.
- **Benchmark tốc độ / baseline**: `v1_window`.
- **Ablation kiểm tra tác động NLI**: `v2_no_nli`.

---

## 4) Lệnh chạy nhanh

- Chạy 1 phiên bản trong full pipeline:
  - `python src/pipeline/main.py --skip-corpus --evidence-variant nli`

- Chạy so sánh 3 phiên bản:
  - `python src/pipeline/main.py --skip-corpus --run-evidence-experiments --evidence-experiment-variants nli,v1_window,no_nli`

- Chạy riêng module so sánh:
  - `python src/pipeline/evidence_experiments.py --input data/corpus/actionability_sentences.parquet --variants nli,v1_window,no_nli`
