# Design — Tăng độ sâu kỹ thuật & EDA cho pipeline ESG-washing (Hướng C)

> Ngày: 2026-06-19 · Nhánh: optimize-research · Trạng thái: đã duyệt brainstorm, chờ review spec

## 1. Bối cảnh & mục tiêu

Pipeline hiện tại (text-based, cross-lingual VI) đã ổn về mô hình: PhoBERT multi-head topic
(E/S/G, masked BCE cho partial labels), PhoBERT commitment, Qwen3 specificity rubric
(anti-hallucination `verify_rubric` + salvage parse), validation thống kê (bootstrap CI,
Friedman/Wilcoxon, Spearman, Kendall tau). Output per (bank, year): `classified.parquet`,
`cti.parquet` (CTI/NAR/QDR), `pillar_shares.parquet`.

**Vấn đề:** tín hiệu washing cuối cùng (CTI/NAR/QDR) là **phép đếm theo rubric**, không phải
mô hình học được → nhìn "thiếu ML/DL/NLP" và EDA mỏng. Mục tiêu: làm bài **chặt chẽ hơn,
show kết quả tốt, nhiều insight** mà **không cần nhãn ngoài** (ràng buộc dữ liệu thực tế).

**Ràng buộc dữ liệu (đã xác nhận với user):**
- KHÔNG có ground-truth washing, KHÔNG có ESG rating ngoài, KHÔNG dùng tài chính/thị trường.
- Trục định lượng **chỉ dùng số tự báo cáo trích từ chính báo cáo** (dư nợ tín dụng xanh,
  phát thải Scope 1/2 vận hành, số cây trồng...).

**Nguyên tắc thiết kế:** mọi tín hiệu washing suy từ bằng chứng văn bản nội tại; **3 nguồn
độc lập đồng quy** (rubric / embedding / số nội tại) thay vì 1 chỉ số.

**Phân tầng N (then chốt cho thống kê):**
- **Chunk-level (N ~ vài nghìn)** = nơi trình diễn ML/DL/NLP thật: classifier benchmark + SHAP,
  embedding signals. Số liệu mạnh, defensible.
- **Panel bank-year (N=30 = 6×5)** = suy luận chỉ số: thống kê non-parametric/bootstrap; model
  ở tầng này chỉ **diễn giải (exploratory)**, KHÔNG tuyên bố dự báo, có caveat N nhỏ tường minh.

## 2. Kiến trúc

```
Reports → chunks.parquet (đã có)
  [L0] Classify: PhoBERT topic + commitment + Qwen3 specificity rubric ............ ĐÃ CÓ
  [L1] EDA giàu (corpus + index) ................. MỚI: experiments/eda_*.py + src/esgwash/eda/style.py
  [L2] Tín hiệu embedding (alignment gap, boilerplate, cluster) ... MỚI: src/esgwash/indices/alignment.py
  [L3] Trục số nội tại (parse figure từ rubric → say-do gap + SHAP) MỚI: src/esgwash/indices/figures_extract.py
  [L4] Model rigor (benchmark + SHAP + cross-lingual) ............. MỞ RỘNG: experiments/eval_models.py
  → Panel hợp nhất: CTI/NAR/QDR + SBS + BRI + say-do gap
```

Mỗi unit có một mục đích, giao tiếp qua parquet/JSON, test độc lập. Không phá vỡ L0 hiện có.

## 3. Khung câu hỏi nghiên cứu

| RQ | Nội dung | Tầng | Trạng thái |
|----|----------|------|-----------|
| RQ1 | Prevalence (CTI/NAR/QDR) | panel | đã có |
| RQ2 | Selective disclosure (E/S/G) | panel | đã có |
| RQ3 | Substance gap / xu hướng theo năm | panel | đã có |
| **RQ4** | **Convergent validity**: tín hiệu embedding (alignment gap, boilerplate) có đồng quy với CTI rubric? | chunk+panel | **mới** |
| **RQ5** | **Say-do gap**: đặc trưng nào "giải thích" specificity (SHAP, chunk-level) + lệch khát-vọng-vs-giao-hàng theo trụ | chunk+panel | **mới** |

## 4. EDA (Phần 2) — style notebook scores-EDA, figure tiếng Anh

**Module dùng chung:** `src/esgwash/eda/style.py` — `PALETTE` (paper `#f7f4ed`, ink `#1f2a33`,
accent `#b55239`, accent2 `#2b7a78`, highlight `#d6a84f`, grid `#c9bfa9`, muted `#5c6770`,
panel `#efe8da`), `SCORE_CMAP` (navy→green→gold→terracotta), `DELTA_CMAP` + `DELTA_NORM`
(`TwoSlopeNorm`), `style_axes()` (title trái bold + subtitle muted, bỏ spine, chỉ grid y).
Mọi figure import từ đây. Output PNG (dpi 140) → `experiments/figures/`, bảng JSON → `experiments/eda/`.

**`experiments/eda_corpus.py`** (đọc `chunks.parquet` + `outputs/cti/*/classified.parquet`):
1. Chunk-size lattice + CDF (phân bố `token_count`, twin-axis CDF, vạch P10/Q1/Median/Q3/P90).
2. Coverage cartography — heatmap bank×year (n_chunk/n_commitment) masked `imshow` + marginal.
3. Label composition — bar E/S/G/commitment positive-rate per bank-year + phân bố `spec_level`
   kèm **entropy + effective states** (`-Σ p log2 p`, `2**H`).
4. Noise-filter retention — bar tỷ lệ bảng/boilerplate đã loại (từ `_cleaning_summary.json`).

**`experiments/eda_indices.py`** (đọc `experiments/panel/panel.csv` + classified):
5. CTI/NAR/QDR inter-quantile ribbons theo năm.
6. CTI cartography bank×year + marginal (mean CTI, volatility) + outline ô `n_commit` thấp.
7. CTI trajectories per bank — `LineCollection` + `DELTA_CMAP` (đỏ=CTI tăng, teal=giảm).
8. Washing-space PCA — `numpy.linalg.eigh` thủ công trên ma trận bank-year chuẩn hoá
   [CTI, NAR, QDR, n_commit, share E/S/G] → biplot + PC loadings. (Bổ sung cột SBS/BRI vào
   biplot ở Phase 4 sau khi Phase 1 tính xong — figure regenerate.)
9. Selective disclosure — E/S/G share heatmap + lệch-so-ngành (`DELTA_CMAP`).

## 5. Method mới (Phần 3) — công thức minh bạch, KHÔNG trọng số bịa

### M1 · Claim–Evidence Alignment Gap (RQ4) — `src/esgwash/indices/alignment.py`
- Embed mọi chunk commitment (tái dùng `src/esgwash/corpus/sentence_embedder.py`).
- Mỗi cam kết **mơ hồ** c (`spec_level=0`) trong (bank,year):
  `backing(c) = max_q cosine(c, q)` với q là chunk **định lượng** (`spec_level=2`) cùng
  (bank,year); nếu không có q → `backing(c)=0`.
- **SBS(bank,year) = mean_c backing(c)** trên các cam kết mơ hồ. SBS thấp = claim mơ hồ
  không có bằng chứng định lượng đỡ → củng cố CTI cao.
- *Chỉ cosine + max + mean — không trọng số học.* RQ4: `Spearman(CTI, 1−SBS)` + scatter.

### M2 · Boilerplate Reuse Index (RQ4) — cùng module
- `boiler(c) = max_{c'} cosine(c, c')` với c' là commitment của **ngân hàng khác cùng năm**.
- **BRI(bank,year) = mean_c boiler(c)**. BRI cao = ngôn ngữ generic dùng lại = cheap-talk.
- KMeans (k chọn bằng silhouette, quét k=3..10) trên embedding commitment + top-words/cluster
  (CountVectorizer, bỏ stopword VI) → "washing themes"; heatmap cluster×bank-year. Kiểm:
  bank CTI cao có dồn vào cluster boilerplate.

### M3 · Trục số nội tại & Say-Do gap (RQ5) — `src/esgwash/indices/figures_extract.py`
- Parse `spec_rubric` JSON đã lưu trong `classified.parquet` → mỗi item định lượng:
  `figure` + `action_or_event` (đã qua `verify_rubric` lọc số bịa → **không cần model trích mới**).
- Phân loại keyword map: {green_credit, emissions, energy, social_donation, trees_count,
  training, other} — map tường minh trong code, không model.
- Bảng feature bank-year: n_commit, n_quantified, n_named, n_vague, cti/nar/qdr, pillar shares,
  SBS, BRI, n_figure theo type.
- **Say-Do gap** per (bank,year,pillar) = `CTI_pillar − QDR_pillar` ∈ [−1,1] (dương = nói nhiều
  hơn làm). Suy từ chỉ số đã có, minh bạch.
- **Model giải thích chunk-level + SHAP**: GradientBoosting dự đoán `spec_level` (0/1/2) từ
  feature cấu trúc chunk (token_count, has_digit, n_digit, pillar probs, commitment prob,
  vị trí trong doc, độ dài câu trung bình...) → **SHAP summary plot** = đặc trưng nào đẩy
  specificity vs mơ hồ. Stratified k-fold CV + macro-F1. Đây là phần ML+explainability hợp lệ
  (N lớn). *Không leak nhãn (không encode bank/ticker — bài học từ notebook SHAP bị data leakage).*
- Panel-level (N=30): correlation matrix toàn feature (`DELTA_CMAP` heatmap) + say-do gap theo
  trụ (bar) — **chỉ mô tả, caveat N nhỏ tường minh, không dự báo**.

**Triangulation:** CTI (rubric) ⟷ 1−SBS / BRI (embedding) ⟷ Say-Do gap (số nội tại).

## 6. Model rigor (Phần 4a) — mở rộng `experiments/eval_models.py`
- Topic & commitment trên gold EN/VI: per-class F1, PR-curve, k-fold CV.
- So sánh encoder cross-lingual: PhoBERT vs XLM-R vs (ClimateBERT cho EN) — train EN → test VI
  (translate-train) → bảng transfer. Củng cố đóng góp cross-lingual (Plan B).
- SHAP/attention attribution token-level trên classifier → figure explainability.
- Rubric validation: agreement LLM-vs-human trên mẫu nhỏ + ablation `verify_rubric`.

## 7. Phasing & outputs

| Phase | Nội dung | Compute | Output |
|-------|----------|---------|--------|
| 0 | `eda/style.py` + `eda_corpus.py` + `eda_indices.py` | CPU | `experiments/figures/*.png`, `experiments/eda/*.json` |
| 1 | M1+M2 `alignment.py` (embedding, cluster) | GPU | cột SBS/BRI vào panel, figure cluster |
| 2 | M3 `figures_extract.py` (parse rubric, say-do, GBM+SHAP) | CPU | bảng feature bank-year, SHAP plot |
| 3 | Model rigor (`eval_models.py`) | GPU | bảng F1/PR/CV/transfer, SHAP classifier |
| 4 | Panel hợp nhất + viết RQ4/RQ5 vào `report.md` | CPU | panel mở rộng + report |

## 8. Ràng buộc (bắt buộc tuân thủ)
- **Figure tiếng Anh** (font + nội dung legend/title) — academic report.
- **Không compile LaTeX** — chỉ quản nội dung paper.
- **Kaggle end-to-end**: bước GPU (embedding M1/M2, eval encoder) phải chạy được TRONG một lần
  chạy pipeline trên Kaggle GPU, KHÔNG tách pha truyền file thủ công. Entry module bootstrap
  `sys.path` (chèn `src/`) để `from esgwash...` chạy khi user `import from src.esgwash...`.
- **Không bịa công thức/trọng số**: SBS/BRI/Say-Do chỉ dùng cosine/max/mean và chỉ số đã có.
- Giữ nguyên L0; mọi thứ mới đọc output đã có, không sửa contract `classified.parquet`.

## 9. Rủi ro & giảm thiểu
- **N=30 panel nhỏ** → mọi model panel-level chỉ exploratory + caveat; ML "thật" đặt ở chunk-level.
- **Data leakage** (bài học notebook SHAP) → tuyệt đối không đưa bank/ticker id vào feature model.
- **Embedder phủ tiếng Việt** → dùng embedder đã có trong repo; nếu yếu, cân nhắc multilingual
  (XLM-R / LaBSE) — quyết định khi đo silhouette/định tính cluster.
- **Parse figure từ rubric** rủi ro thấp (đã có `verify_rubric`); type-map keyword cần QA mẫu.
