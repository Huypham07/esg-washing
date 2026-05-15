# ESG-Washing Risk Analyzer

## Tổng quan

Pipeline nhận đầu vào là báo cáo thường niên dạng PDF hoặc văn bản thuần, sau đó:

1. Trích xuất văn bản (OCR qua Docling nếu cần)
2. Phân loại chủ đề ESG (E / S\_labor / S\_community / S\_product / G)
3. Phân loại mức độ hành động (Implemented / Planning / Indeterminate)
4. Liên kết bằng chứng hỗ trợ tuyên bố ESG (TF-IDF + NLI cross-lingual)
5. Tính chỉ số **EWRI** (ESG Washing Risk Index) ∈ [0, 100]

Hai classifier (topic + action) được finetune từ PhoBERT với **Neuro-Symbolic AI Type 4**: tri thức ký hiệu (GRI, Bloom Taxonomy, Hyland Metadiscourse) được biên dịch thành semantic loss trong quá trình huấn luyện.

## Cấu trúc thư mục

```
esg-washing/
├── config/
│   ├── pipeline.yml          # Tham số pipeline
│   └── train.yml             # Tham số huấn luyện
├── data/
│   ├── extracted/            # Văn bản thô / ZIP source PDFs
│   └── labels/               # Dữ liệu gán nhãn (topic, action)
└── src/
    ├── pipeline/
    │   ├── run.py            # CLI
    │   ├── pipeline.py       # ESGWashingPipeline
    │   ├── ewri.py           # Tính EWRI
    │   └── ...
    └── training/             # Huấn luyện mô hình phân loại
```

## Hướng dẫn chạy

### Setup môi trường

Python 3.10+, cài dependencies:

```bash
pip install -r requirements.txt
```

---

### 1. Demo — phân tích 1 báo cáo thường niên

```bash
python -m src.pipeline.run \
  --input path/to/annual_report.pdf \
  --output outputs/demo \
  --bank "TEN_NGAN_HANG" \
  --year 2024
```

| Tham số | Mô tả | Mặc định |
|---|---|---|
| `--input` | PDF hoặc `.txt` | (bắt buộc) |
| `--output` | Thư mục kết quả | `outputs/demo` |
| `--bank` | Tên ngân hàng | `DEMO_BANK` |
| `--year` | Năm báo cáo | `2024` |

Kết quả:

```
outputs/demo/
├── extracted.txt        # Văn bản sau OCR/cleaning
├── report.html          # Báo cáo HTML
└── enriched.parquet     # DataFrame đầy đủ
```
---

### 2. Huấn luyện model

> **Pre-trained models** đã publish trên HuggingFace, tự động tải về khi chạy pipeline:
> - [`huypham71/esg-topic-classifier`](https://huggingface.co/huypham71/esg-topic-classifier)
> - [`huypham71/esg-action-classifier`](https://huggingface.co/huypham71/esg-action-classifier)

Để finetune lại từ đầu, chạy `notebooks/train-model.ipynb` trên Colab (GPU T4+), hoặc:

```bash
# Neuro-Symbolic (với semantic loss)
python -m src.training.train_model --task topic
python -m src.training.train_model --task action

# Baseline
python -m src.training.train_model --task topic  --no-neuro-symbolic
python -m src.training.train_model --task action --no-neuro-symbolic
```

Model được lưu vào `outputs/models/{topic,action}_classifier/final/`.

#### Tinh chỉnh siêu tham số (Optuna)

```bash
python -m src.training.tune_hyperparams --task topic  --n-trials 50
python -m src.training.tune_hyperparams --task action --n-trials 50
```

---

### 3. Liên kết bằng chứng + Grid Search tham số EWRI

#### 3a. Liên kết bằng chứng (Evidence Linking)
```bash
python -m src.pipeline.evidence_experiments \
  --input  data/corpus/actionability_sentences.parquet \
  --output-dir outputs/experiments/evidence
```

| Biến thể | Mô tả |
|---|---|
| `nli` | TF-IDF + semantic similarity + NLI (mDeBERTa) — **dùng cho RQ3** |
| `window` | Cửa sổ lân cận ± 5 câu |
| `no_nli` | TF-IDF + semantic, không NLI |

Kết quả: `outputs/experiments/evidence/evidence_{nli,window,no_nli}.parquet`

Để chỉ chạy biến thể `nli`:
```bash
python -m src.pipeline.evidence_experiments --variants nli
```

#### 3b. Grid Search tham số EWRI

```bash
python -m src.pipeline.ewri_grid_search \
  --input       outputs/experiments/evidence/evidence_nli.parquet \
  --output      outputs/ewri_grid_search_results.csv \
```

Kết quả:
- `outputs/ewri_grid_search_results.csv` — toàn bộ tổ hợp

Sau khi tìm được tham số tối ưu, cập nhật vào `config/pipeline.yml`:

```yaml
ewri:
  action_penalty:
    Implemented:   <P_Impl>
    Planning:      <P_Plan>
    Indeterminate: <P_Indet>
  evidence_sensitivity:
    Implemented:   <L_Impl>
    Planning:      <L_Plan>
    Indeterminate: <L_Indet>
  contradiction_amplifier: <C>
```

---

## Cấu hình

### `config/pipeline.yml` — các mục quan trọng

```yaml
model:
  topic:
    hf_model_id: "huypham71/esg-topic-classifier"
    path: "outputs/models/topic_classifier/final"   # local override
    max_length: 128

  nli_model: "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

ewri:
  action_penalty:          # P(y)
    Implemented:   0.02
    Planning:      0.15
    Indeterminate: 0.55
  evidence_sensitivity:    # λ(y)
    Implemented:   1.00
    Planning:      0.85
    Indeterminate: 0.50
  contradiction_amplifier: 1.8   # C khi NLI = contradiction
```