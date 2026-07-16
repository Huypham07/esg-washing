# Data layout

```
data/
  extracted/raw_ocr_annual_report.zip   # nguồn corpus duy nhất (59+ txt OCR, 11 bank dirs)
  source_dataset/                       # gold gốc đã công bố — chỉ đọc, không sửa
    topic/        environmental_2k, social_2k, governance_2k (EN, ESGBERT)
    subst/        specificity, commitments_actions, env_claims, action_500 (EN, ClimateBERT/ESGBERT)
                  netzero_reduction.csv: GIỮ trên đĩa nhưng KHÔNG dùng (quyết định 2026-06-10)
    ml_promise/   Trainset_*.json (gốc) + ml_promise.csv (đã flatten: en/fr/ja/zh, 1346 dòng)
  translate/                            # bản dịch máy sang VI, cùng tên file với source
                                        # ml_promise_vi.csv sẽ thêm vào đây khi dịch xong
  legacy/                               # artifact cũ — KHÔNG dùng làm input pipeline mới
    corpus/       parquet dẫn xuất cũ (build lại từ extracted/)
    labels/       nhãn LLM cũ (chỉ được dùng weak-check phân phối)
  processed/                            # output pipeline (generated, gitignore)
    sentences.parquet, blocks.parquet, gold/*.parquet
  annotation/                           # file giao user gán nhãn (generated)
```

Chi tiết nguồn gốc + license từng dataset: `source_dataset/README.md`.
Quy trình xử lý: `superpowers/specs/01-data.md`.
