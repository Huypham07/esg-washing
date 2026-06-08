# Phương án B — Đề cương triển khai

> **ESG-washing assessment cho NHTM Việt Nam bằng cross-lingual transfer (translate-train).**
> Nguyên tắc: mọi thành phần đo lường **dựa trên công bố đã có** (citation + data), **không trọng số tự bịa**, **không gán nhãn câu thủ công**, **không train trên nhãn do LLM sinh**.

---

## 0. Tóm tắt

- **Đối tượng phân tích:** corpus đã có — 10 NHTM VN × 2020–2024, **119.733 câu** (`data/corpus/sentences.parquet`). Dùng **chỉ để inference + chạy pipeline**, không làm nguồn nhãn train.
- **Nguồn nhãn train:** **gold tiếng Anh đã công bố** (ESGBERT/ClimateBERT, đã tải về `data/en_gold/`) → **dịch sang tiếng Việt bằng `google/translategemma-4b-it`** (giữ nguyên nhãn) → fine-tune **PhoBERT**.
- **3 đầu ra mô hình:** (1) **Topic** E/S/G (multi-label phẳng; non-ESG = cả ba = 0); (2) **Substantiveness** ordinal L0–L3; (3) **Grounding** câu–bằng chứng bằng entailment.
- **Chỉ số washing:** **decoupling-as-residual** — hồi quy Substantiation (S) theo Talk (T), phần dư = mức washing (Bromley & Powell 2012). Không trọng số tay.
- **Validation 4 tầng, 0 gán nhãn:** (i) gold EN; (ii) consistency transfer EN→VN; (iii) synthetic manipulation + known-group VN; (iv) case study.
- **Venue mục tiêu:** ESWA / Knowledge-Based Systems / workshop ClimateNLP–FinNLP.

---

## 1. Câu hỏi nghiên cứu, giả thuyết & đóng góp

### 1.1 Research questions
- **RQ1.** Translate-train EN→VN cho topic + substantiveness đạt mức nào trên gold EN (upper bound) và giữ được bao nhiêu khi chuyển sang VN (đo bằng consistency, không cần gold VN)?
- **RQ2.** Grounding + claim–evidence entailment cải thiện **faithfulness** của phán đoán substantiveness bao nhiêu so với không grounding?
- **RQ3.** Chỉ số decoupling có **construct/criterion validity** với tín hiệu công khai miễn phí của NH VN (VNSI, báo cáo bền vững riêng, assurance bên thứ ba, công bố tín dụng xanh) không?
- **RQ4.** Lọc nhiễu (confident learning) có cải thiện mô hình khi train trên gold đã dịch máy (dịch có thể làm câu lệch khỏi nhãn gốc) không?

### 1.2 Giả thuyết
- **H1.** Translate-train > zero-shot XLM-R trên consistency VN. [Exp-3]
- **H2.** Pipeline có grounding có faithfulness cao hơn (ít "claim-without-evidence"), dù Macro-F1 substantiveness có thể không đổi. [Exp-4]
- **H3.** Nhóm "weak signal" (không VNSI/assurance/tín dụng xanh) có **washing-residual cao hơn** nhóm "strong signal". [Exp-6]
- **H4 (null-friendly).** Confident learning cải thiện Macro-F1/QWK so với train thẳng trên bản dịch nhiễu; nếu không → báo cáo thẳng. [Exp-5]

### 1.3 Đóng góp
1. **Cross-lingual transfer ESG-washing EN→VN** đầu tiên cho NH VN, full E/S/G (translate-train PhoBERT + confident learning xử lý nhiễu dịch).
2. **Khung faithfulness có thể đo** cho phán đoán washing (grounding + entailment giữ phân phối NLI).
3. **Giao thức validation không-gán-nhãn 4 tầng** + bộ **translate-train ESG tiếng Việt** công khai.

---

## 2. Kho dữ liệu

### 2.1 Corpus phân tích VN (đối tượng — đã có, chỉ inference)
| File | Đơn vị | Số dòng | Cột chính |
|---|---|---|---|
| `data/corpus/sentences.parquet` | câu | 119.733 | doc_id, bank, year, section_id, block_id, sent_id, sentence, ctx_prev, ctx_next, block_type, section_title |
| `data/corpus/blocks.parquet` | block | 99.444 | doc_id, bank, year, section_title, block_text, block_type, source_path |

> Corpus từ OCR → cần làm sạch (Bước 1). `ctx_prev/ctx_next` dùng cho grounding ngữ cảnh. Không train trên corpus.

### 2.2 Gold tiếng Anh (`data/en_gold/`) — nguồn nhãn train qua translate-train
**Topic (sentence-level):**
| File | Dòng | Nhãn | Citation |
|---|---|---|---|
| `topic/environmental_2k.csv` | 2000 | `env` ∈ {0,1} | Mehra et al. 2022 (ESGBERT) |
| `topic/social_2k.csv` | 2000 | `soc` ∈ {0,1} | — |
| `topic/governance_2k.csv` | 2000 | `gov` ∈ {0,1} | — |
| `topic/nature_2200.csv` | 2200 | dùng cột `Nature` ∈ {0,1} (⇒ thêm mẫu E dương) | — |

**Substantiveness / claim / action:**
| File | Dòng (train/test) | Nhãn | Citation |
|---|---|---|---|
| `subst/specificity.{train,test}.parquet` | 1000/320 | specific ∈ {0,1} | Bingler et al. 2022 |
| `subst/commitments_actions.{train,test}.parquet` | 1000/320 | commitment/action ∈ {0,1} | ClimateBERT |
| `subst/action_500.csv` | 500 | `action` ∈ {0,1} | ESGBERT |
| `subst/netzero_reduction.csv` | 3441 | `target` ∈ {none, reduction, net-zero} | Schimanski et al. 2023 |
| `subst/env_claims.{train,val,test}.parquet` | 2117/265/265 | claim ∈ {0,1} | Stammbach et al. 2022 |
| `subst/detection.{train,test}.parquet` | 1300/400 | climate-related ∈ {0,1} | climatebert/climate_detection |
| `subst/tcfd.{train,test}.parquet` | 1300/400 | 5-class TCFD | climatebert/tcfd_recommendations |

> Khi load: bỏ cột `Unnamed: 0`, chuẩn hoá unicode NFC (vài file có U+2212/U+FFFD).

### 2.3 Tín hiệu "known-group" VN (`data/external/bank_signals.csv` — cần tạo, mức tổ chức, KHÔNG phải nhãn câu)
10 bank × 5 năm: `vnsi_member`, `standalone_sr`, `gri_referenced`, `external_assurance`, `green_credit_disclosed`.

---

## 3. Phương pháp luận

### 3.1 Định nghĩa washing
Decoupling giữa **talk** (mức phát ngôn ESG) và **substantiation** (mức chứng minh thực chất) — biểu tượng vs thực chất. Cơ sở: Meyer & Rowan 1977; Bromley & Powell 2012. Vận hành NLP: Bingler 2022; Schimanski 2023.

### 3.2 Cross-lingual transfer: translate-train
- **Lý do (không zero-shot thuần):** Hu et al. 2020 (XTREME) — translate-train thường > zero-shot khi MT tốt. Encoder: **PhoBERT** (Nguyen & Nguyen 2020); **XLM-R** (Conneau et al. 2020) làm baseline zero-shot.
- **MT engine:** `google/translategemma-4b-it` (Gemma-3 4B instruction-tuned cho dịch; gated *manual*).
- **Quy trình:** dịch gold EN → VN, giữ nhãn, fine-tune PhoBERT. Rủi ro: dịch làm câu lệch nhãn (vd mất con số ⇒ "specific" hoá "vague") → kiểm soát ở Bước 4.3 (QE + confident learning).

### 3.3 Grounding & faithfulness
Khung FEVER (Thorne et al. 2018) / CLIMATE-FEVER (Diggelmann et al. 2020), gồm 3 lớp (không dùng LLM):
1. **Dense retrieval** span bằng chứng trong cùng (bank, year) bằng bi-encoder `bkai-foundation-models/vietnamese-bi-encoder` (đo liên quan claim↔evidence).
2. **Claim–evidence NLI** bằng `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` → **giữ phân phối entail/neutral/contradict**, không nhị phân hoá.
3. **Khớp slot định lượng** (IE): trích số/target/baseline/năm từ claim, kiểm có con số đỡ tương ứng trong evidence — lớp hợp ESG nhất, giải thích được. (Tuỳ chọn: faithfulness kiểu QA/QuestEval với ViT5.)
- Faithfulness đo bằng: % câu substantive **có** evidence entail; "claim-without-evidence rate"; span F1/IoU so với gold synthetic.

### 3.4 Chỉ số washing (không trọng số bịa)
- Chuẩn hoá z-score toàn corpus cho **T** và **S**.
- **washing = phần dư hồi quy S ~ T** (decoupling-as-residual). Residual dương cao = nói nhiều, chứng minh ít. Hệ số do dữ liệu quyết định (OLS/robust).
- Tổng hợp theo (bank, year, trụ E/S/G); báo cáo CI bootstrap.
- Không có nhánh "walk thật" (CDP/MSCI rỗng cho VN) — nêu rõ giới hạn.

---

## 4. Pipeline chi tiết từng bước

### Bước 0 — Môi trường
- Cài thiếu: `pip install datasets accelerate sentencepiece`.
- **MT:** `google/translategemma-4b-it` qua `transformers` — gated *manual*: accept license trên HF + `huggingface-cli login`. Cần GPU (~8–10GB, bf16/4-bit).
- `underthesea` đã có — PhoBERT **bắt buộc tách từ** trước tokenize.

### Bước 1 — Làm sạch corpus VN  → `src/training/corpus/clean_corpus.py`
1. NFC + sửa lỗi OCR (ký tự rác, ghép dòng), bỏ câu < 4 token / toàn số / mục lục.
2. Khử trùng lặp gần (MinHash) — boilerplate lặp giữa các năm.
3. Word-segment (`underthesea`) → cột `sentence_ws`.
4. Output: `data/corpus/sentences_clean.parquet`.

### Bước 2 — Dịch gold EN→VN  → `src/training/translate/translate_gold.py`
1. Dịch cột `text` mọi bộ `data/en_gold/` sang VN bằng `translategemma-4b-it`, **giữ nhãn**. Lưu `data/vi_gold/<task>/...parquet` (cột `text_en, text_vi, label`).
2. Prompt theo format instruct (EN→VI); batch + bf16/4-bit.
3. Word-segment `text_vi` → `text_vi_ws`.

### Bước 3 — Harmonize nhãn  → `src/training/labeling/harmonize_labels.py`
- **Topic:** 3 nhãn phẳng E/S/G; thêm câu `Nature=1` vào tập E-dương.
- **Substantiveness:** dựng thang ordinal §5.2 từ specificity + commitments_actions + netzero.

### Bước 4 — Huấn luyện encoder
- **4.1 Topic (multi-label phẳng):** PhoBERT + 3 đầu sigmoid độc lập (E/S/G), loss BCE. Train trên `vi_gold/topic`. → `src/training/train_topic.py`
- **4.2 Substantiveness (ordinal):** PhoBERT + đầu ordinal (CORN/CORAL) trên thang §5.2. → `src/training/train_substantiveness.py`
- **4.3 QE & lọc nhiễu:** round-trip VN→EN agreement + COMET-QE (lọc câu dịch xấu); **Confident Learning** (Northcutt 2021) — tìm & loại câu lệch nhãn–văn bản sau dịch (data-centric, model-agnostic). → `src/training/denoise.py`

### Bước 5 — Grounding & entailment  → `src/pipeline/grounding.py` (nâng từ `evidence_*` + `nli_verifier.py`)
1. Mỗi câu claim/commitment (substantiveness ≥ ngưỡng): retrieval span trong cùng (bank, year) bằng bi-encoder `bkai-foundation-models/vietnamese-bi-encoder` (+ câu lân cận, BM25 bổ trợ).
2. **Claim–evidence NLI** `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`: phân phối {entail, neutral, contradict}, giữ nguyên.
3. **Khớp slot định lượng** (IE): KPI/số, baseline, target/mốc, assurance — kiểm con số trong claim có evidence đỡ.
4. Output: `outputs/grounding/<bank>_<year>.parquet`.

### Bước 6 — Talk/Substantiation & washing index  → `src/pipeline/decoupling.py` (thay `ewri.py`)
- **T:** tỉ lệ câu ESG topic-positive (chuẩn hoá theo độ dài báo cáo).
- **S:** tổng hợp substantiveness ordinal × tín hiệu grounding (tỉ lệ claim có evidence entail). Chuẩn hoá z hoặc factor đầu PCA (báo cáo loadings), không trọng số tay.
- **washing = residual(S ~ T)** theo (bank, year, trụ). Bootstrap CI; ranking.

### Bước 7 — Báo cáo & case study  → mở rộng `src/pipeline/demo_report.py`
- Bảng xếp hạng washing-residual; trích câu "high-talk low-substantiation".

---

## 5. Thiết kế nhãn

### 5.1 Topic — phẳng, multi-label
3 nhãn nhị phân độc lập; non-ESG = cả ba = 0.

| Nhãn | Nguồn gold |
|---|---|
| **E** | `environmental_2k` + câu `Nature=1` của `nature_2200` |
| **S** | `social_2k` |
| **G** | `governance_2k` |

> Partial-label: 3 bộ gold chủ yếu là câu khác nhau → train 3 binary độc lập, mỗi cái dùng trọn ~2k, không ép ma trận multi-label đầy đủ.

### 5.2 Substantiveness — thang ordinal 4 mức
Cơ sở: Bingler 2022 (specificity) + Schimanski 2023 (target).

| Mức | Tên | Định nghĩa vận hành |
|---|---|---|
| L0 | Non-claim / boilerplate | commitments_actions=0 & env_claims=0 |
| L1 | Symbolic / vague | là claim nhưng specificity=0, không target |
| L2 | Specific-no-target | specificity=1 nhưng netzero=none |
| L3 | Substantive | specificity=1 ∧ target∈{reduction,net-zero} ∧ có slot định lượng/baseline |

---

## 6. Metric đánh giá

### 6.1 Mô hình (gold EN test — upper bound)
- **Topic (multi-label):** Macro-F1, per-class F1/P/R, micro-F1, subset accuracy.
- **Substantiveness ordinal:** QWK, Macro-F1, MAE ordinal.
- **NLI:** Macro-F1 3 lớp; calibration (ECE).

### 6.2 Transfer EN→VN (không gold VN)
- **Back-translation consistency:** acc(model_VI(x_vi) == model_EN(x_vi→en)).
- **Teacher–student agreement:** Cohen's κ giữa teacher EN (trên bản dịch) và PhoBERT VN.

### 6.3 Grounding / faithfulness
- % claim có evidence entail; claim-without-evidence rate; span F1/IoU (vs gold synthetic); faithfulness delta có/không grounding.

### 6.4 Chỉ số washing
- **Known-group:** Mann–Whitney U + rank-biserial giữa strong/weak signal.
- **Criterion/convergent:** Spearman ρ với (vnsi_member, gri_referenced, external_assurance, green_credit_disclosed) + lexicon mơ hồ độc lập.
- **Synthetic sensitivity:** chỉ số tăng khi xoá bằng chứng khỏi câu substantive.
- **Stability:** Kendall τ ranking giữa biến thể pipeline.

---

## 7. Thực nghiệm & ablation

| # | Tên | Mục đích | Setup | Metric |
|---|---|---|---|---|
| **Exp-1** | Baselines | model không chỉ học từ vựng | majority, TF-IDF+LogReg, LLM few-shot (gold EN test) | Macro-F1, QWK |
| **Exp-2** | Upper-bound EN | trần năng lực | fine-tune gold EN, test EN | §6.1 |
| **Exp-3** | Ablation transfer (RQ1,H1) | giá trị translate-train | zero-shot XLM-R vs translate-train PhoBERT | §6.2 |
| **Exp-4** | Ablation grounding (RQ2,H2) | faithfulness | pipeline ± claim–evidence entailment | §6.3 |
| **Exp-5** | Ablation lọc nhiễu (RQ4,H4) | giá trị confident learning | train thẳng vs +Confident Learning | Macro-F1, QWK |
| **Exp-6** | Known-group (RQ3,H3) | construct validity | residual theo strong/weak signal | Mann–Whitney U, ρ |
| **Exp-7** | Synthetic manipulation | ground-truth tự tạo | xoá/chèn bằng chứng (CheckList, Ribeiro 2020) | Δ index, span F1 |
| **Exp-8** | Sensitivity/robustness | ranking bền vững | đổi ngưỡng, encoder (PhoBERT/XLM-R/ViDeBERTa); adversarial boilerplate | Kendall τ |
| **Exp-9** | Case study | face validity | trích high-talk/low-substantiation | định tính |

Thứ tự: Exp-2 → 1 → 3 → 5 → 4 → 7 → 6 → 8 → 9.

---

## 8. Thay đổi code

**Bỏ:** `src/pipeline/ewri.py`, `ewri_grid_search.py`, nhị phân hoá `es_combined`, `src/training/neuro_symbolic.py`, dùng `topic_llm_labeler` làm nhãn train.

**Giữ/sửa:** `src/pipeline/evidence_*` + `nli_verifier.py` → span retrieval + entailment giữ phân phối NLI.

**Thêm:** `clean_corpus.py`, `translate_gold.py`, `harmonize_labels.py`, `denoise.py`, `train_topic.py`, `train_substantiveness.py`, `grounding.py`, `decoupling.py`, `src/eval/{eval_gold,transfer_consistency,synthetic_manipulation,known_group}.py`.

---

## 9. Lộ trình (~12 tuần)
1. **T1–2:** clean corpus + dịch gold (Bước 1–2) + `bank_signals.csv`.
2. **T3–4:** harmonize nhãn + train topic (Exp-2) + baselines (Exp-1).
3. **T5–6:** train substantiveness + transfer ablation (Exp-3) + lọc nhiễu (Exp-5).
4. **T7–8:** grounding (Bước 5) + ablation grounding (Exp-4).
5. **T9:** decoupling index (Bước 6) + synthetic (Exp-7).
6. **T10:** known-group (Exp-6) + sensitivity (Exp-8).
7. **T11–12:** case study, viết bài, đóng gói dataset.

---

## 10. Rủi ro & giảm thiểu
- **Chất lượng dịch:** QE + back-translation + confident learning; báo cáo độ giảm so với EN upper-bound.
- **Substantiation full-ESG mỏng (gold chủ yếu climate/E):** S cho S/G yếu hơn E → báo cáo per-trụ, nêu rõ giới hạn.
- **Không có walk thật:** nêu giới hạn; bù bằng known-group + synthetic + convergent.
- **OCR noise:** làm sạch + khử trùng lặp trước khi đo T.

## 11. Tài liệu tham chiếu
Meyer & Rowan 1977; Bromley & Powell 2012 · Conneau et al. 2020 (XLM-R) · Hu et al. 2020 (XTREME) · Nguyen & Nguyen 2020 (PhoBERT) · Bingler et al. 2022 (Cheap Talk, FRL) · Schimanski et al. 2023 (ClimateBERT-NetZero, EMNLP) · Stammbach et al. 2022 (environmental_claims) · Thorne et al. 2018 (FEVER) · Diggelmann et al. 2020 (CLIMATE-FEVER) · Northcutt et al. 2021 (Confident Learning) · Ribeiro et al. 2020 (CheckList) · Mehra et al. 2022 (ESGBERT) · Greenwashing-NLP survey arXiv 2502.07541 (2025).
