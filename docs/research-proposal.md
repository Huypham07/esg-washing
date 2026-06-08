# ESG-Washing Assessment — Đề cương nghiên cứu (2 phương án)

> Nguyên tắc xuyên suốt: **mọi thành phần áp dụng/nâng cấp đều phải dựa trên một công bố đã có** (ghi rõ citation + data cần dùng). Không có công thức tự bịa, không có trọng số tự chọn.

---

## 0. Bối cảnh & vấn đề (chung cho cả 2 phương án)

**Định nghĩa ESG-washing dùng trong bài (chính danh, có lý thuyết):** *decoupling* giữa lời nói (talk) và hành động/chứng minh (walk) — tức công bố mang tính **biểu tượng (symbolic)** thay vì **thực chất (substantive)**.
Cơ sở: Meyer & Rowan (1977); Bromley & Powell (2012, *ceremonial vs substantive adoption*). Vận hành đo bằng NLP: Bingler, Kraus, Leippold, Webersinke (2022, *Cheap Talk and Cherry-Picking*, Finance Research Letters); Schimanski et al. (2023, ClimateBERT-NetZero, EMNLP).

**Khe trống tận dụng (gap):**
1. NLP washing hầu hết chỉ làm **green/climate (trụ E)**; **ESG-washing toàn trụ E/S/G** còn ít — survey Greenwashing-NLP (arXiv 2502.07541, 2025).
2. Không tồn tại "verified greenwashing ground truth" → mọi chỉ số phải đo bằng **proxy có cấu trúc + validate construct**, không phải tự đặt trọng số (survey 2502.07541, mục Validation gaps).
3. **Faithfulness/grounding** của phán đoán washing chưa được đo nghiêm túc → đây là novelty kỹ thuật.

**Điều bỏ hẳn so với bản hiện tại (`src/pipeline/ewri.py`, `ewri_grid_search.py`):**
- Công thức `WRS = P(y)(1−λ(y)·support)·C` và toàn bộ trọng số `P, λ, C` (tự đặt → loại).
- Grid-search tối đa hóa Kendall's W (self-referential, không phải validation → loại).
- `es_combined` nhị phân hóa NLI (vứt thông tin → loại).
- Schema action tự chế `Implemented/Planning/Indeterminate` (không gắn công bố nào → thay bằng schema có nhãn expert, xem §A/B).

---

## 1. Neuro-Symbolic: làm đúng nghĩa là gì, hiện đang sai chỗ nào, cần data gì

Bạn nghi ngờ đúng. Phải tách bạch 3 họ phương pháp NS đã công bố:

| Họ NS | Công bố gốc | Ràng buộc trên cái gì | Symbolic knowledge lấy từ đâu |
|---|---|---|---|
| **Semantic Loss** | Xu et al. 2018 (ICML) | **biến đầu ra** (exactly-one, mutual-exclusion, cấu trúc) | tiên đề logic về nhãn (vd: mỗi câu đúng 1 nhãn) |
| **Semantic-based Regularization / Logic Tensor Networks** | Diligenti et al. 2017 (AIJ); Badreddine et al. 2022 (AIJ, LTN) | **luật fuzzy nối predicate(input) → label** | luật miền độc lập với nhãn |
| **Hierarchy-consistency** | Giunchiglia & Lukasiewicz 2020 (NeurIPS, C-HMCNN) | **nhất quán cha–con trong taxonomy** | cây taxonomy (vd GRI/ESG) |

**Bản hiện tại (`src/training/neuro_symbolic.py`) sai ở đâu:**
- Phần `exactly_one_loss` (WMC) → **đúng** theo Xu et al. 2018, giữ lại.
- Phần `implication_loss`/`negation_loss`: predicate là **regex keyword** (`grounded_rules.py`) chạy trên **chính câu input**, mà các keyword này **gần trùng** với tín hiệu LLM (`topic_llm_labeler.py`) đã dùng để sinh nhãn → **rò rỉ vòng lặp (label leakage)**: tri thức ký hiệu = hàm sinh nhãn. Kết quả "cải thiện" là giả. **Đây là lỗi chí tử reviewer sẽ bắt.**

**Dùng NS cho đúng (data lấy ở đâu):**
1. **Hierarchy-consistency** (Giunchiglia & Lukasiewicz 2020): symbolic = **cây taxonomy GRI/ESG** (vd taxonomy 3/16/119 lớp của `cea-list-ia/ESG-classification-en`, vốn xấp xỉ GRI). Ràng buộc: P(child) ≤ P(parent). **Knowledge độc lập hoàn toàn với nhãn → không rò rỉ.** Data cần: chỉ cần cây taxonomy, không cần thêm câu.
2. **Rubric-logic của substantiveness** (Diligenti et al. 2017 / LTN): luật kiểu *"L3-Substantive ⇒ has_quantitative ∧ has_target ∧ has_baseline"*. Predicate **không** lấy từ regex trên câu, mà từ **các slot bằng chứng do module grounding phát hiện** (KPI, baseline, target, assurance). Knowledge = **định nghĩa rubric GRI**, độc lập với nhãn lớp. Data cần: output của module grounding (§ pipeline bước 3).
3. **Đo lợi ích trung thực:** mọi ràng buộc NS phải có **ablation vs no-NS** trên **gold tiếng Anh** (§ thực nghiệm). Nếu không cải thiện → báo cáo thẳng, không ép.

> Kết luận NS: giữ semantic loss cho hierarchy/exactly-one (Xu 2018 + Giunchiglia 2020), bỏ phần implication-từ-keyword. NS **không phải đóng góp chính** mà là một thành phần có ablation.

---

## PHƯƠNG ÁN A — Toàn bộ tiếng Anh (English end-to-end)

### A.1 Mục tiêu & RQ
- **RQ1:** Phân loại topic (GRI) + substantiveness có grounding đạt mức nào trên gold đã công bố?
- **RQ2:** Grounding bắt buộc + claim–evidence entailment cải thiện **faithfulness** bao nhiêu (so với không grounding)?
- **RQ3:** Chỉ số decoupling (talk-vs-walk) có **construct/criterion validity** với dữ liệu hiệu suất thực bên ngoài không?

### A.2 Dữ liệu
- **Corpus phân tích:** báo cáo thường niên/bền vững **tiếng Anh** của ngân hàng/doanh nghiệp niêm yết (công khai, tải được; vd ngân hàng toàn cầu). *Phải gom mới — đây là chi phí chính của phương án A.*
- **Train/test topic:** ESGBERT `environmental/social/governance_2k` (dataset publish), FinBERT-ESG-9 & `cea-list-ia/ESG-classification-en` (model publish → dùng làm teacher), Mehra et al. 2022 (ESGBERT, arXiv 2203.16788).
- **Train/test substantiveness:** ML-Promise (Chen et al., EMNLP 2025: promise→evidence→clarity{Clear/NotClear/**Misleading**}→timing), ClimateBERT `commitments_actions`/`specificity` (Bingler/Schimanski), `netzero-reduction` (Schimanski 2023).
- **External "walk" (KHẢ THI ở phương án A):** CDP emissions, LSEG/Refinitiv hoặc MSCI ESG, RepRisk controversies, Net Zero Tracker. → cho phép đo **talk-vs-walk thật**.

### A.3 Pipeline NLP (mỗi bước gắn công bố)
1. **ESG topic classification** → fine-tune encoder trên gold ESG, harmonize về **taxonomy GRI** (xương sống = 16-class cea-list). NS hierarchy-consistency (Giunchiglia & Lukasiewicz 2020).
2. **Claim/commitment & substantiveness** → schema ML-Promise (Chen 2025) + specificity (Bingler 2022).
3. **Evidence grounding + verification** → truy hồi span bằng chứng rồi **claim–evidence entailment** theo khung FEVER (Thorne et al. 2018) / CLIMATE-FEVER (Diggelmann et al. 2020); đo **% grounding hợp lệ, span F1/IoU, faithfulness**. Không nhị phân hóa NLI — giữ phân phối entail/neutral/contradict.
4. **Đo Talk (T) & Substantiation (S)** theo §0 (operationalize Bingler 2022).
5. **Chỉ số washing = decoupling** (xem A.4).

### A.4 Đo washing (không trọng số bịa)
- T, S z-score toàn corpus; **washing = phần dư hồi quy S theo T** (decoupling-as-residual; cơ sở Bromley & Powell 2012). Hệ số do dữ liệu quyết định.
- Bổ sung khả thi ở A: hồi quy **S (text) theo walk thật (emissions/rating)** → đo decoupling talk-vs-walk đúng nghĩa.

### A.5 Validation
- **RQ1/RQ2:** trên **test split gold đã công bố** (Macro-F1, QWK ordinal — Cohen; Krippendorff α báo cáo sẵn của dataset).
- **RQ3 criterion validity:** tương quan chỉ số với CDP/rating/controversies (Spearman).
- **Synthetic manipulation / behavioral test** (Ribeiro et al. 2020, CheckList): xóa bằng chứng khỏi câu substantive → kiểm tra chỉ số phản ứng đúng; robustness adversarial boilerplate (survey 2502.07541 yêu cầu).

### A.6 Điểm mạnh / rủi ro / khả thi
- **Mạnh:** validation mạnh nhất (có walk thật), benchmark được, **0 gán nhãn mới**.
- **Rủi ro:** đông đối thủ; **mất domain VN**; phải gom corpus English mới.
- **Khả thi triển khai: CAO** (dữ liệu + model + external đều sẵn).

---

## PHƯƠNG ÁN B — Toàn bộ tiếng Việt (Vietnamese end-to-end, cross-lingual transfer)

### B.1 Mục tiêu & RQ
- **RQ1:** Transfer EN→VN cho topic + substantiveness đạt mức nào (đo bằng consistency, không cần gold VN)?
- **RQ2:** Grounding + entailment cải thiện faithfulness bao nhiêu?
- **RQ3:** Chỉ số decoupling có construct validity với **tín hiệu công khai miễn phí** của NH VN không?

### B.2 Dữ liệu
- **Corpus phân tích:** 10 NHTM VN × 2020–2024 **đã có sẵn** (`data/corpus/`). Đây là asset + novelty.
- **Train (nhãn expert, hết lệch):** **translate-train** — dịch máy dữ liệu gold tiếng Anh (§A.2) sang tiếng Việt rồi fine-tune **PhoBERT** (Nguyen & Nguyen 2020). Cơ sở translate-train: Hu et al. 2020 (XTREME); encoder đa ngữ XLM-R: Conneau et al. 2020.
- **Bổ sung silver:** đồng thuận đa-LLM (chỉ giữ câu Gemini+GPT+Claude đồng ý) làm silver, công khai gọi là silver; xử lý nhãn nhiễu bằng **confident learning** (Northcutt et al. 2021) hoặc **co-teaching** (Han et al. 2018).
- **External walk:** đánh giá khả thi = **KHÔNG khả thi** (CDP/MSCI/RepRisk coverage VN ~rỗng). → không dựa vào.

### B.3 Pipeline NLP
Giống A.3 về kiến trúc, khác ở **cross-lingual**: encoder PhoBERT/ViDeBERTa; teacher English (cea-list/FinBERT-9) gán silver trên bản dịch; NS thêm vai trò **bắc cầu domain/văn hóa khi transfer** (tri thức GRI + quy định SBV — Chỉ thị 03/2015 tín dụng xanh, Nghị định 13/2023 dữ liệu cá nhân), áp đúng theo §1 (knowledge độc lập nhãn).

### B.4 Đo washing
Như §0/A.4 nhưng **bỏ nhánh walk thật** (không có data). Chỉ dùng **decoupling-as-residual nội văn bản** (S theo T).

### B.5 Validation KHÔNG cần chuyên gia gán nhãn (4 tầng)
1. **Mô hình:** trên **gold tiếng Anh có sẵn** (Macro-F1, QWK) — 0 công gán nhãn.
2. **Transfer EN→VN:** back-translation consistency (VN→EN→model English) + teacher–student agreement; cơ sở pseudo-label/consistency: Hu et al. 2020.
3. **Chỉ số washing:**
   - **Synthetic manipulation** (Ribeiro et al. 2020): ground-truth tự tạo bằng xóa/thêm bằng chứng → 0 gán nhãn.
   - **Known-group validity** bằng tín hiệu công khai: thành viên **VNSI (HOSE Sustainability Index)**; có báo cáo bền vững riêng / tham chiếu GRI / assurance bên thứ ba (trích từ chính báo cáo); tỉ lệ tín dụng xanh (bank nào công bố). Kiểm định nhóm, 0 gán nhãn câu.
   - **Convergent validity:** tương quan với đo lường text độc lập (specificity classifier, lexicon mơ hồ).
4. **Face validity:** case study định tính.

### B.6 Điểm mạnh / rủi ro / khả thi
- **Mạnh:** novelty cao nhất (ESG-washing NLP đầu tiên cho NH VN, full E/S/G, low-resource cross-lingual); tận dụng asset; **0 gán nhãn thủ công**.
- **Rủi ro:** không có external walk → validation yếu hơn A (bù bằng 4 tầng trên); shift kép (ngôn ngữ+domain) cần đo cẩn thận; venue nên nhắm **ESWA/Knowledge-Based Systems / workshop ClimateNLP–FinNLP** hơn main ACL.
- **Khả thi triển khai: TRUNG BÌNH–CAO** (dữ liệu sẵn, không cần annotate; nút thắt là chất lượng transfer ở phần substantiation full-ESG — data English mỏng ~600–1.3k).

---

## 2. Thực nghiệm & ablation (chung)
- **Baseline bắt buộc** (survey 2502.07541 yêu cầu): majority, **keyword/TF-IDF**, **LLM few-shot** — để chứng minh model không chỉ học từ vựng.
- **Ablation NS:** no-NS vs +hierarchy vs +rubric-logic, trên gold EN.
- **Ablation grounding:** có/không claim–evidence entailment → đo faithfulness.
- **(B) Ablation transfer:** zero-shot vs translate-train vs +silver-đa-LLM.
- **Sensitivity:** thứ hạng washing bền với biến thể pipeline.
- **Metrics:** Macro-F1, QWK (ordinal), Krippendorff α; grounding: %valid-evidence, span F1/IoU; index: Spearman với proxy/known-group.

## 3. So sánh & khuyến nghị
| | A. English | B. Vietnamese (cross-lingual) |
|---|---|---|
| Validation | mạnh (walk thật) | trung bình (4 tầng, không walk) |
| Novelty | thấp–trung | **cao** |
| Annotate mới | 0 | 0 |
| Corpus | phải gom mới | **đã có** |
| Khả thi | cao | trung–cao |
| Venue | có thể nhắm cao hơn | ESWA/KBS/workshop |

**Khuyến nghị: Phương án B** nếu ưu tiên novelty + tận dụng asset (và chấp nhận venue tầm trung); **Phương án A** nếu ưu tiên sức mạnh validation và sẵn sàng gom corpus English.

## 4. Việc cần làm với code hiện tại
- **Bỏ:** `ewri.py` (công thức WRS, P/λ/C), `ewri_grid_search.py`, `es_combined` nhị phân, phần `implication_loss`/`negation_loss` từ keyword trong `neuro_symbolic.py`, nhãn `topic_llm_labeler` làm nhãn train chính.
- **Giữ/sửa:** `exactly_one_loss` (Xu 2018); thêm hierarchy-consistency (Giunchiglia 2020); module grounding nâng từ TF-IDF→span retrieval + entailment giữ phân phối NLI.
- **Thêm:** loader các dataset English gold; pipeline translate-train (B); module đo T/S + decoupling residual; bộ test synthetic-manipulation + known-group.

## 5. Tài liệu tham chiếu chính
Meyer & Rowan 1977; Bromley & Powell 2012 · Xu et al. 2018 (ICML, Semantic Loss) · Diligenti et al. 2017 (AIJ) · Badreddine et al. 2022 (AIJ, LTN) · Giunchiglia & Lukasiewicz 2020 (NeurIPS, C-HMCNN) · Conneau et al. 2020 (XLM-R) · Hu et al. 2020 (XTREME) · Nguyen & Nguyen 2020 (PhoBERT) · Bingler et al. 2022 (Cheap Talk, FRL) · Schimanski et al. 2023 (ClimateBERT-NetZero, EMNLP) · Chen et al. 2025 (ML-Promise, EMNLP) · Thorne et al. 2018 (FEVER) · Diggelmann et al. 2020 (CLIMATE-FEVER) · Northcutt et al. 2021 (Confident Learning) · Han et al. 2018 (Co-teaching) · Ribeiro et al. 2020 (CheckList) · Zheng et al. 2023 (LLM-as-Judge) · Mehra et al. 2022 (ESGBERT) · Greenwashing-NLP survey arXiv 2502.07541 (2025).
