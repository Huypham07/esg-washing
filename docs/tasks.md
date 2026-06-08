# Tasks — Phương án B (bám sát `research-plan-B.md`)

> Quy ước: `[ ]` chưa làm · `[~]` đang làm · `[x]` xong & đã verify. Mỗi task lớn có **Deliverable** (file/output) và **Done when** (tiêu chí verify). Cột "Plan" trỏ về mục trong đề cương. Ta verify từng task trước khi đánh `[x]`.

Thứ tự phụ thuộc: **T0 → T1, T2 (song song) → T3 → T4 → T5 → T6 → T7 → T8 → T9 → T10**. T12 (dọn code) làm sớm khi tiện.

---

## T0 — Môi trường & truy cập  · Plan §4 Bước 0
- [ ] T0.1 Tạo/kích hoạt venv; `pip install datasets accelerate sentencepiece` (đã có torch/transformers/underthesea/pandas).
- [ ] T0.2 Accept license `google/translategemma-4b-it` trên HF + `huggingface-cli login` (token).
- [ ] T0.3 Smoke-test dịch 1 câu EN→VI bằng TranslateGemma (xác nhận GPU + bf16/4-bit chạy).
- [ ] T0.4 Tải/kiểm model NLI `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` + bi-encoder `bkai-foundation-models/vietnamese-bi-encoder` (đã chốt).
- **Deliverable:** `requirements.txt` cập nhật; ghi chú model ids vào `docs/tasks.md` (mục Notes cuối file).
- **Done when:** dịch thử ra tiếng Việt hợp lý; không lỗi auth/OOM.

## T1 — Làm sạch corpus VN  · Plan §4 Bước 1 → `src/training/corpus/clean_corpus.py`
- [ ] T1.1 Load `data/corpus/sentences.parquet`; NFC normalize; sửa lỗi OCR phổ biến (ký tự rác, ghép dòng vỡ).
- [ ] T1.2 Lọc câu < 4 token / toàn số / dòng mục lục / header-footer lặp.
- [ ] T1.3 Khử trùng lặp gần (MinHash/LSH) trong và giữa các năm.
- [ ] T1.4 Word-segment (`underthesea`) → cột `sentence_ws`.
- [ ] T1.5 Thống kê trước/sau (số câu, % loại bỏ) → log.
- **Deliverable:** `data/corpus/sentences_clean.parquet` + bảng thống kê.
- **Done when:** spot-check 30 câu ngẫu nhiên sạch & tách từ đúng; phân bố bank/year hợp lý.

## T2 — Dịch gold EN→VN (translate-train)  · Plan §4 Bước 2 → `src/training/translate/translate_gold.py`
- [ ] T2.1 Loader chuẩn hoá mọi bộ `data/en_gold/` (bỏ `Unnamed: 0`, NFC, hợp nhất schema `text,label,task,split`).
- [ ] T2.2 Hàm dịch batch bằng TranslateGemma (prompt EN→VI instruct, giữ nhãn).
- [ ] T2.3 Dịch toàn bộ gold; lưu `data/vi_gold/<task>/<split>.parquet` (`text_en,text_vi,label`).
- [ ] T2.4 Word-segment `text_vi` → `text_vi_ws`.
- [ ] T2.5 Spot-check chất lượng dịch (≥30 câu/đầu task).
- **Deliverable:** thư mục `data/vi_gold/` đầy đủ topic + subst.
- **Done when:** đủ số dòng khớp gold gốc; dịch giữ nghĩa & con số; nhãn không xê dịch.

## T3 — Harmonize nhãn  · Plan §4 Bước 3, §5 → `src/training/labeling/harmonize_labels.py`
- [ ] T3.1 Topic: gộp env/soc/gov thành 3 cột nhị phân; thêm `Nature=1` của nature_2200 vào tập E-dương; lưu split train/val/test.
- [ ] T3.2 Substantiveness: dựng thang ordinal L0–L3 (§5.2) từ specificity + commitments_actions + netzero; định nghĩa rule rõ ràng, log phân bố mức.
- [ ] T3.3 Tạo train/val/test cho từng task (giữ split gốc của gold khi có).
- **Deliverable:** `data/vi_gold/topic/{train,val,test}.parquet`, `data/vi_gold/subst_ordinal/{train,val,test}.parquet`.
- **Done when:** phân bố nhãn hợp lý; rule ordinal tái lập được; không leak giữa split.

## T4 — Train Topic classifier  · Plan §4 Bước 4.1, §6.1 → `src/training/train_topic.py`
- [ ] T4.1 Dataset/collator PhoBERT (input `text_vi_ws`), 3 đầu sigmoid độc lập, loss BCE.
- [ ] T4.2 Train (3 binary riêng hoặc 1 thân 3 đầu — thử cả hai).
- [ ] T4.3 Eval trên **gold EN test** (sau khi dịch test? hoặc test EN trực tiếp bằng XLM-R baseline) → chốt cách đo upper-bound (Exp-2).
- [ ] T4.4 Lưu checkpoint + metrics (Macro-F1, per-class, micro-F1, subset acc).
- **Deliverable:** `outputs/models/topic/`, `outputs/metrics/topic.json`.
- **Done when:** Macro-F1 vượt baseline Exp-1; số liệu lưu lại tái lập được.

## T5 — Train Substantiveness classifier (ordinal)  · Plan §4 Bước 4.2, §6.1 → `src/training/train_substantiveness.py`
- [ ] T5.1 Đầu ordinal (CORN/CORAL) trên PhoBERT.
- [ ] T5.2 Train trên `subst_ordinal`.
- [ ] T5.3 Eval: QWK, Macro-F1, MAE ordinal.
- [ ] T5.4 Lưu checkpoint + metrics.
- **Deliverable:** `outputs/models/subst/`, `outputs/metrics/subst.json`.
- **Done when:** QWK > baseline; MAE hợp lý; lưu lại.

## T6 — Lọc nhiễu + đánh giá transfer  · Plan §4 Bước 4.3, §6.2, Exp-3/Exp-5
- [ ] T6.1 QE: round-trip VN→EN agreement + COMET-QE → cờ câu dịch xấu. → `src/training/denoise.py`
- [ ] T6.2 Confident Learning (Northcutt) trên train; sinh tập đã lọc + danh sách câu nghi lệch nhãn.
- [ ] T6.3 **Exp-5:** train thẳng vs +Confident Learning → so Macro-F1/QWK.
- [ ] T6.4 **Exp-3:** zero-shot XLM-R vs translate-train PhoBERT (back-translation consistency, teacher–student κ). → `src/eval/transfer_consistency.py`
- **Deliverable:** `outputs/metrics/exp3.json`, `outputs/metrics/exp5.json`.
- **Done when:** bảng so sánh đầy đủ; kết luận H1/H4 (kể cả null).

## T7 — Grounding & entailment  · Plan §4 Bước 5, §3.3 → `src/pipeline/grounding.py`
- [ ] T7.1 Retriever span ứng viên trong cùng (bank, year): câu lân cận + BM25/dense.
- [ ] T7.2 Slot detector: KPI định lượng, baseline, target/mốc, assurance.
- [ ] T7.3 Claim–evidence NLI (model T0.4) → phân phối {entail,neutral,contradict}, giữ nguyên.
- [ ] T7.4 Chạy trên câu claim của corpus → `outputs/grounding/<bank>_<year>.parquet`.
- [ ] T7.5 **Exp-4:** pipeline ± entailment → faithfulness delta. → `src/eval/` faithfulness.
- **Deliverable:** `outputs/grounding/*`, `outputs/metrics/exp4.json`.
- **Done when:** span hợp lệ; phân phối NLI không bị nhị phân hoá; Exp-4 có số.

## T8 — Chỉ số washing (decoupling)  · Plan §4 Bước 6, §3.4 → `src/pipeline/decoupling.py`
- [ ] T8.1 Tính **T** (mật độ ESG topic-positive, chuẩn hoá độ dài) theo (bank,year,trụ).
- [ ] T8.2 Tính **S** (substantiveness ordinal × grounding), z-score / PCA factor (báo cáo loadings).
- [ ] T8.3 Hồi quy S~T → residual = washing; bootstrap CI; ranking.
- **Deliverable:** `outputs/washing/index.parquet` + ranking.
- **Done when:** index có CI; không dùng trọng số tay; tái lập được.

## T9 — Validation chỉ số  · Plan §5? §6.4, Exp-6/Exp-7/Exp-8
- [ ] T9.1 Tạo `data/external/bank_signals.csv` (VNSI/SR/GRI/assurance/green-credit). · Plan §2.3
- [ ] T9.2 **Exp-6 known-group:** Mann–Whitney U + ρ giữa residual và signals. → `src/eval/known_group.py`
- [ ] T9.3 **Exp-7 synthetic:** xoá/chèn bằng chứng → kiểm index đúng hướng + span F1. → `src/eval/synthetic_manipulation.py`
- [ ] T9.4 **Exp-8 sensitivity:** đổi ngưỡng/encoder (PhoBERT/XLM-R/ViDeBERTa) → Kendall τ ranking.
- **Deliverable:** `outputs/metrics/exp6.json`, `exp7.json`, `exp8.json`.
- **Done when:** H3 có kết luận; index phản ứng đúng với synthetic; ranking ổn định.

## T10 — Baselines, case study, đóng gói  · Plan Exp-1/Exp-2/Exp-9, §1.3
- [ ] T10.1 **Exp-1 baselines:** majority, TF-IDF+LogReg, LLM few-shot trên gold EN test.
- [ ] T10.2 **Exp-2 upper-bound EN** (chốt cùng T4.3/T5.3).
- [ ] T10.3 **Exp-9 case study:** trích high-talk/low-substantiation theo bank → `src/pipeline/demo_report.py`.
- [ ] T10.4 Đóng gói dataset translate-train công khai (`data/vi_gold/` + README + license).
- **Deliverable:** `outputs/metrics/exp1.json`, `exp2.json`; báo cáo case study; bộ dữ liệu công khai.
- **Done when:** đủ bảng cho paper; dataset có README + nguồn gốc rõ.

## T11 — Viết bài
- [ ] T11.1 Ghép kết quả tất cả Exp vào bảng + hình (dùng `docs/figures/architecture.png`).
- [ ] T11.2 Viết Limitations (no walk thật, S/G mỏng, dịch máy).

## T12 — Dọn code cũ  · Plan §8 (làm sớm khi tiện)
- [ ] T12.1 Xoá `src/pipeline/ewri.py`, `ewri_grid_search.py`, `src/training/neuro_symbolic.py`.
- [ ] T12.2 Gỡ nhị phân hoá `es_combined`; gỡ dùng `topic_llm_labeler` làm nhãn train.
- [ ] T12.3 Cập nhật `config/pipeline.yml` / `run.py` theo pipeline mới.
- **Done when:** repo không còn tham chiếu module đã bỏ; pipeline mới chạy end-to-end trên 1 bank thử.

---

## Notes (đã chốt)
- **MT model:** `google/translategemma-4b-it`
- **Encoder VN:** `vinai/phobert-base`
- **NLI:** `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` (chính); `presencesw/phobert-large-vinli-3-label` (VN-specific, chỉ để so ở Exp-8).
- **Retrieval bằng chứng:** `bkai-foundation-models/vietnamese-bi-encoder` (bi-encoder).
- **Verify khâu grounding = 3 lớp:** dense retrieval + NLI (giữ phân phối) + khớp slot định lượng. (QA/QuestEval-ViT5 = optional.)
- **Lọc nhiễu:** chỉ **Confident Learning** (data-centric), bỏ Co-teaching.
- **Upper-bound EN (Exp-2):** GIỮ, nhưng chỉ là mốc tham chiếu transfer-gap (độc lập với lọc nhiễu); cách đo: fine-tune model trên gold EN gốc, test EN gốc.
