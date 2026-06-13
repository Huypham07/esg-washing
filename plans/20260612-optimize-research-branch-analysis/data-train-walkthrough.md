# Walkthrough: pipeline data → train của đồng nghiệp (`src/esgwash`)

> Giải thích chi tiết từ bước đầu (raw OCR) đến sau khi train xong M1/M2. Đọc kèm `analysis.md`.
> Phạm vi: Phase A (data) + Phase B (train) — phần ĐÃ code thật. Sau đó classify/ground/index = stub.

## Cách chạy (toàn bộ chỉ 2 nhóm lệnh)

```bash
# PHASE A — dựng data (4 stage tuần tự)
python -m esgwash.pipeline.run --stage all      # build_corpus → prepare_gold → cross_label → export_annotation

# PHASE B — train (đọc bảng nhãn ở data/processed/gold/)
python scripts/train_topic.py --tune 30         # M1: tune Optuna 30 trial → train 5-seed
python scripts/train_claim.py --tune 30         # M2: tương tự, multi-task
```

Mọi stage = **hàm thuần đọc parquet → ghi parquet**, chạy độc lập, config ở `configs/<stage>.yml`.

---

## PHẦN 1 — DATA (Phase A)

### Stage 1 · `build_corpus` (corpus_builder + cleaning + segmentation)
**Vào:** `data/extracted/raw_ocr_annual_report.zip` (59 txt OCR markdown-ish từ docling, tên file `…/<bank>/<…year…>.txt`).
**Làm:**
1. Regex tách `bank, year` từ tên file.
2. `clean_raw_text`: NFC normalize + sửa ký tự OCR (`−–—`→`-`, bỏ `�`, soft-hyphen), **nối từ bị gãy dòng** (`cam-\nkết`→`camkết`), bỏ tag `<image>`.
3. **Bỏ "frequent lines"**: dòng lặp ≥8 lần trong cùng doc (tên bank, "BÁO CÁO THƯỜNG NIÊN" — header/footer).
4. Tách **block** theo dòng trống → phân loại `block_type` ∈ {heading, table, list, paragraph}:
   - heading → mở **section** mới (section_id, section_title).
   - table → mỗi hàng `| a | b |` → text "a | b"; list → bỏ bullet; paragraph → `underthesea.sent_tokenize`.
5. `is_valid_sentence`: giữ câu ≥10 ký tự & ≥3 từ.
6. Sinh `ctx_prev`/`ctx_next` (câu liền kề) cho mỗi câu.
7. `dedup_per_doc`: exact (lowercased) + **minhash LSH near-dup** (16 band, threshold 0.85) **trong cùng doc_id**.

**Ra:** `data/processed/sentences.parquet` — schema giàu: `doc_id, bank, year, section_id, block_id, sent_id, sentence, ctx_prev, ctx_next, block_type, section_title` (+ `blocks.parquet`, `corpus_stats.json`).
**Bẫy quan trọng:** **KHÔNG xoá số liệu/đơn vị** — chúng là tín hiệu specificity & evidence pool cho grounding sau này. `block_type` được giữ để Phase C lọc evidence (table/list = nơi có số).

### Stage 2 · `prepare_gold` (gold_loader → topic_merge + claim_merge)
**`gold_loader`** — nạp EN gốc (`source_dataset/`) + bản dịch VI (`translate/`), **align theo CHỈ SỐ DÒNG** (`assert len(tr)==len(src)`): file dịch cùng thứ tự dòng → nhãn EN map sang VI **không cần infer lại**. Giữ `text_en` để QE-filter/merge.

**`topic_merge.build_topic_table`** (3 tập nhị phân → 1 bảng masked multi-label):
1. Mỗi tập (`environmental/social/governance_2k`) chỉ có nhãn **1 trụ**.
2. `_resolve_dups`: cùng `text_en` mà nhãn mâu thuẫn → bỏ; còn lại giữ dòng đầu.
3. **Merge outer theo `text_en`** → 1 dòng/1 text duy nhất, cột `env/soc/gov` mà **2 cột là NaN** (trụ chưa gán). ⇒ "bảng masked".
4. `split_stratified` 80/10/10 trên **text duy nhất** (chống leak), stratify theo *pattern nhãn* (combo "1-0-NaN"…, hiếm → "rare").
5. `_append_env_claims`: câu positive của env_claims → thêm `env=1` (soc/gov NaN, **train-only**).
   **Ra:** `topic_masked.parquet`.

**`claim_merge.build_claim_table`** (M2):
1. `load_claim_pair`: commitment & specificity **dùng đúng cùng text** → merge theo `text_en` → `[text, commitment, specificity, split]` (bảng multi-task).
2. `_carve_dev`: tách 10% train → dev (stratify combo commitment×specificity).
3. `augment_action`: `action_500` → `commitment=action, specificity=NaN`, train-only (vá khoảng trống cam kết S/G).
4. `augment_ml_promise`: `ml_promise_vi` (EN+FR+JA) → `commitment=promise, specificity=NaN`, train-only.
   **Ra:** `claim_table.parquet ★` (★ = bảng train dùng trực tiếp).

### Stage 3 · `cross_label` (esgbert_labels) — **bước thông minh nhất**
**Vấn đề:** bảng masked còn nhiều NaN (mỗi câu chỉ có nhãn 1 trụ). Masked-BCE xử lý được nhưng mất tín hiệu.
**Cách giải:** dùng **3 classifier ĐÃ CÔNG BỐ** `ESGBERT/{Environmental,Social,Governance}BERT` (Schimanski 2023 — chính model train trên 3 tập 2k gốc) để **điền các ô NaN**:
- `esgbert_probs`: với trụ p, chạy ESGBERT-p trên `text_en`, **chỉ ở ô (split=train AND p là NaN)**. → **chống leak**: model trụ p chưa từng thấy nhãn p của câu đến từ tập khác.
- `fill_cross_labels`: điền 1.0 nếu prob ≥ τ (0.9), 0.0 nếu ≤ 1-τ (0.1), còn lại để NaN. **Nhãn gốc không bị ghi đè.**
- Nhãn topic bất biến qua dịch → map sang VI bằng row-alignment (không infer trên text VI).
**Ra:** `topic_labeled.parquet ★` (+ `esgbert_probs.parquet` cache, `cross_label_stats.json`).
**Tại sao hay:** tránh **confirmation bias** của self-training (không tự train 3 model rồi tự gán nhãn cho chính mình); model EN-native, rẻ (chỉ inference). Ablation: `--masked-only` dùng bảng trước cross-label.

### Stage 4 · `export_annotation` (vn_eval_set)
Lấy ~300 câu corpus **stratified theo bank × trụ-dự-đoán** → `data/annotation/vn_eval_todo.csv` để **người gán** (env/soc/gov/commitment/specificity), đo **Cohen κ**. ⇒ đây là **test set THẬT** (gold EN chỉ là upper-bound).

---

## PHẦN 2 — TRAIN (Phase B)

### Động cơ chung · `trainer.py` (dùng chung cho CẢ M1 và M2)
- **`MultiHeadClassifier`**: PhoBERT backbone + N **đầu sigmoid độc lập** (`Linear(hidden,1)`) trên token `[CLS]`. M1 = 3 đầu (env/soc/gov), M2 = 2 đầu (commitment/specificity) — **cùng 1 class, chỉ khác config heads**.
- **`masked_bce_loss`**: `BCEWithLogitsLoss` nhưng **mask ô NaN** → chỉ tính loss trên ô có nhãn. `pos_weight` per-head (theo tỉ lệ nhãn) chống imbalance. ⇒ đây là **partial-label learning**, mấu chốt cho cả topic lẫn claim.
- **`fit`**: word-segment (`underthesea`) **TRƯỚC** tokenize (bắt buộc PhoBERT, `use_fast=False`); AdamW + linear warmup; grad-clip 1.0; mỗi epoch eval macro-F1 trên dev, **giữ checkpoint best-epoch** (= early stopping ngầm); `epoch_callback` cho Optuna prune.
- **`tune_thresholds`**: sau train, mỗi head quét ngưỡng 0.05–0.95 **tối đa F1 trên dev** — KHÔNG mặc định 0.5.
- **`multi_seed`**: train **5 seed [42-46]**, báo cáo **mean±std** trên test (ngưỡng từ dev), lưu model seed-0.

### Tuning · `tuning.py` (Optuna)
- **TPE sampler** (Bayesian, multivariate) + **MedianPruner** (cắt sớm trial tệ qua epoch_callback).
- Không gian: `lr [1e-5,5e-5] log`, `batch {16,32}`, `warmup [0,0.2]`, `weight_decay [1e-3,0.1] log`, `dropout [0.05,0.3]`.
- **KHÔNG tune epochs** (cố định max_epochs=10; best-epoch checkpoint lo early-stop).
- Study lưu **SQLite → resumable** (chạy thêm trial không mất lịch sử).
- Tune trên **dev, 1 seed**; **KHÔNG chạm test** khi tune. `save_best_params`→json; `apply_best_params`→config cuối.

### M1 · `topic_model.py`
`TopicModel(MultiHeadTrainer)`; `predict()` → 3 prob + nhãn đã threshold + **trụ chính** (`idxmax`, = `non_esg` nếu không trụ nào vượt ngưỡng → loại khỏi tầng sau).

### M2 · `claim_model.py`
`ClaimModel` multi-task 2 đầu. `run_single_task_fallback`: train **2 model 1-đầu riêng** để so trên dev (giữ cả 2 cho ablation). Aux head env_claims **đã bỏ** (2026-06-12).

### Baselines · `baselines.py`
- `tfidf_lr_baseline`: TF-IDF (1-2gram, 50k feat) + LogisticRegression per head — **sàn phải vượt**.
- `xlmr_zero_shot`: train trên **EN gốc**, infer thẳng VI (không dịch, không word-segment) — nhánh đối chứng của translate-train trong ma trận transfer.

### Entry scripts
- `train_topic.py [--tune N] [--lang vi|en]` → `outputs/models/topic/vi/` + `outputs/metrics/topic_vi.json`. `--lang en` = upper-bound (roberta-base, train+test EN).
- `train_claim.py [--tune N] [--single-task]` → `outputs/models/claim/vi/` + `claim_vi.json`.

---

## Điểm dừng (sau train)

`stages.py` đăng ký `classify / ground / index` = `_todo` → **`raise NotImplementedError`**. Nên: model M1/M2 đã có, nhưng **chưa có gì chạy chúng trên corpus** → chưa có CTI/grounded-CTI/validation. Đây đúng là chỗ track local của tôi đã đi xa hơn.

---

## 8 điều đáng học (để cải tiến track của tôi)

| # | Kỹ thuật đồng nghiệp | Track tôi hiện tại | Học gì |
|---|---|---|---|
| 1 | **Masked-BCE 1 model** partial-label (NaN mask) | 3 model nhị phân riêng | Gộp về 1 encoder chia sẻ, đỡ trùng lặp; hoặc ít nhất học cách mask NaN |
| 2 | **ESGBERT cross-label** điền NaN (anti-leak, anti-confirmation-bias) | silver LLM gán hết | Dùng model công bố thay LLM tự-gán → bảo vệ trước hội đồng |
| 3 | **Multi-task claim** (commit+spec 1 encoder, vì trùng 100% text) | train rời | Gộp → nhất quán biểu diễn, 1 lượt inference |
| 4 | **Align dịch theo dòng + assert** (map nhãn EN→VI, no drift) | silver gán trực tiếp VI | Translate-train giữ nhãn gốc người-gán |
| 5 | **Threshold tuning per-head** trên dev | (kiểm tra lại track tôi) | Không để mặc định 0.5 |
| 6 | **Optuna TPE+MedianPruner**, epochs-không-tune, SQLite-resume, test-bất-khả-xâm | 5 Optuna study đã có | Đảm bảo KHÔNG chạm test khi tune; prune để rẻ |
| 7 | **multi_seed 5 seed mean±std** | (kiểm tra) | Báo cáo độ lệch → rigor cho paper |
| 8 | **Schema corpus giàu** (block_type/section/ctx) | corpus đơn giản hơn | Thêm block_type → phục vụ grounding + fix artifact bsc |

## Câu hỏi mở
1. Track tôi đã có threshold-tuning per-head + multi-seed chưa? (cần đối chiếu `src/training` local).
2. `ml_promise_vi.csv` (1200 mẫu dịch) — tôi có bản dịch đó không, hay phải tự dịch?
3. ESGBERT cross-label có cần GPU/được phép tải model gated không? (3 model ESGBERT public, không gated).
