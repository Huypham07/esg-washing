# ESG-Washing — đo cheap talk trong báo cáo ngân hàng Việt Nam

Hệ thống NLP đo **khoảng cách giữa cam kết và độ thực chất** (talk vs substance) trong báo cáo
thường niên của 10 NHTM Việt Nam, đủ ba trụ E/S/G. Vì Việt Nam không có "walk-data" bên thứ ba,
độ thực chất được đo **ngay trong văn bản** qua độ cụ thể của cam kết (specificity), không dùng
grounding nội văn bản (vòng tròn — đã gỡ, xem `legacy/`).

Thiết kế & lý do: `docs/superpowers/specs/2026-06-16-esg-washing-pipeline-redesign-design.md`.

## Chỉ số

Trên mỗi ô (ngân hàng, năm, trụ), denominator = cam kết có gắn trụ ESG (gate bằng topic):

| Chỉ số | Định nghĩa | Ý nghĩa |
|---|---|---|
| **CTI** | tỉ lệ cam kết Mức 0 (mơ hồ) | cheap talk / washing |
| **NAR** | tỉ lệ Mức 1 (hành động có tên, chưa định lượng) | vùng xám |
| **QDR** | tỉ lệ Mức 2 (định lượng, quy về chủ thể) | thực chất |

CTI + NAR + QDR = 1, mỗi chỉ số kèm bootstrap CI. Kèm **selective disclosure** (phân bố câu ESG
theo trụ — đo né trụ khó). Mọi chỉ số chỉ là tỉ lệ output classifier, không tự đặt thang/trọng số.

## Pipeline

```
build_chunks  : OCR sạch -> tách câu + lọc nhiễu -> semantic chunk (bi-encoder, ≤256 token)
run           : Topic E/S/G (PhoBERT) -> Commitment (PhoBERT) -> Specificity (LLM-rubric 3 mức)
                -> CTI/NAR/QDR + selective disclosure + bootstrap CI
```

Pipeline chạy **end-to-end** trong một lệnh. Bước specificity dùng LLM nên nặng → chạy cả repo
trên Kaggle GPU rồi mang `outputs/` về.

Mô hình (tự tải từ HuggingFace khi chạy):
- Topic E/S/G: [`huypham71/esg-topic`](https://huggingface.co/huypham71/esg-topic)
- Commitment: [`dqa2412/esg-washing-optimized`](https://huggingface.co/dqa2412/esg-washing-optimized)
- Specificity: `Qwen/Qwen3-1.7B` (cấu hình `configs/specificity.yml`)

## Chạy

```bash
pip install -r requirements.txt          # cần Java (JAVA_HOME) cho word-seg VnCoreNLP

# 1) Build corpus semantic chunk (1 lần) -> data/chunks.parquet
python -m esgwash.corpus.build_chunks

# 2) Chạy pipeline end-to-end (nặng -> Kaggle GPU). 1 ngân hàng-năm hoặc toàn bộ:
python -m esgwash.run --bank bidv --year 2023
python -m esgwash.run --all

# 3) Phân tích/biểu đồ từ outputs (tuỳ chọn)
python -m experiments.analyse
```

Kết quả mỗi (bank, year) tại `outputs/cti/<bank>/<year>/`:
- `classified.parquet` — chunk + topic/commitment/specificity
- `cti.parquet` — CTI/NAR/QDR + CI + selective disclosure theo trụ
- `info_check.json` — chẩn đoán (tỉ lệ non-ESG, phân bố spec_level, parse fail)

## Cấu trúc

```
configs/        # cấu hình từng bước (chunk, topic, commitment, specificity, index, validation)
src/esgwash/
  corpus/       # build_chunks, semantic_split, sentence_embedder
  models/       # topic, commitment, specificity (LLM-rubric)
  indices/      # cti (CTI/NAR/QDR), disclosure, bootstrap
  validation/   # known_group, synthetic, sensitivity, digit_shortcut, runner (audit)
  run.py        # orchestrator end-to-end
experiments/    # analyse, baselines, eval
legacy/         # grounding đã gỡ khỏi luồng chính (vòng tròn)
docs/           # đề cương + specs/plans
```

## Kiểm thử

```bash
python -m pytest tests/ -q
```
