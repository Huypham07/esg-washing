# Plan 01 — Kế hoạch triển khai code

Bám theo specs 00–05. Kiến trúc codebase: **package `src/esgwash/` theo layered
pipeline architecture** (chuẩn phổ biến cho ML research code — tách data/models/
pipelines/evaluation, config-driven, mọi stage là hàm thuần đọc/ghi artifact parquet,
chạy được độc lập và tái lập bằng config + seed).

## 0. Cấu trúc thư mục đích

```
configs/                      # YAML per stage (single source of truth)
  corpus.yml  topic.yml  claim.yml  grounding.yml  index.yml  validation.yml
src/esgwash/
  config.py                   # load + validate config (dataclass)
  data/                       # Spec 01
    corpus_builder.py         #   raw zip → blocks → sentences (P1)
    cleaning.py               #   NFC, OCR fix, noise filter, dedup
    gold_loader.py            #   load en_gold + translate, chuẩn hoá cột
    topic_merge.py            #   masked-label table + cross-pseudo-label
    noise_filter.py           #   QE (cometkiwi/LaBSE) + Confident Learning
    vn_eval_set.py            #   sampling câu cho human annotation + load nhãn
  nlp/
    segmentation.py           #   word-segment (pyvi/VnCoreNLP) — bắt buộc cho PhoBERT
  models/                     # Spec 02
    topic_model.py            #   M1: PhoBERT 3-sigmoid, masked BCE
    claim_model.py            #   M2: multi-task commitment+specificity (+aux env_claims)
    baselines.py              #   TF-IDF+LR, zero-shot XLM-R
    trainer.py                #   vòng train chung (seed, pos_weight, threshold tuning)
  grounding/                  # Spec 03
    evidence_pool.py          #   candidate pool rules (số liệu/table/specific-fact)
    retriever.py              #   bkai bi-encoder top-k per doc
    nli.py                    #   mDeBERTa XNLI, giữ phân phối
    support.py                #   support = max P_entail; grounded@θ
  indices/                    # Spec 04
    cti.py                    #   CTI, grounded-CTI
    disclosure.py             #   selective disclosure share
    bootstrap.py              #   percentile bootstrap CI, min-n pooling
  validation/                 # Spec 05
    known_group.py            #   V1 Mann-Whitney + effect size
    synthetic.py              #   V2 perturbation
    sensitivity.py            #   V3 θ-sweep, Kendall τ
  pipeline/
    stages.py                 #   đăng ký stage: build_corpus → classify → ground → index
    run.py                    #   CLI: python -m esgwash.pipeline.run --stage all|<name>
scripts/                      # entrypoint mỏng: train_topic.py, train_claim.py, ...
tests/                        # unit test cho logic thuần (merge, pool rules, cti, bootstrap)
outputs/{models,metrics,grounding,index}/
```

Code cũ `src/pipeline/`, `src/training/` giữ nguyên đến khi pipeline mới chạy end-to-end
1 bank (T12 mới xoá: ewri*, neuro_symbolic, es_combined, topic_llm_labeler).

## 1. Phase A — Nền tảng dữ liệu (T0, T1, T3)

| Bước | Module | Deliverable | Done-when |
|---|---|---|---|
| A1 | configs + esgwash skeleton + requirements | repo import được | `python -m esgwash.pipeline.run --help` |
| A2 | corpus_builder: unzip raw → blocks/sentences (tái dùng logic document_loader cũ) | `data/processed/sentences.parquet` | spot-check 30 câu; thống kê bank/year |
| A3 | gold_loader + topic_merge (masked table 6k dòng) | `data/processed/gold/*.parquet` | phân bố nhãn khớp spec 01; không leak split |
| A4 | vn_eval_set sampling 300 câu | `data/annotation/vn_eval_todo.csv` | giao user gán nhãn (song song các phase sau) |

## 2. Phase B — Train models (T4, T5, T6)

| Bước | Module | Deliverable | Done-when |
|---|---|---|---|
| B1 | trainer + baselines | metrics/exp1.json | TF-IDF+LR chạy đủ 3 task |
| B2 | M1 topic masked-BCE (+ threshold tuning) | outputs/models/topic | Macro-F1 > baseline trên dev dịch |
| B3 | M2 claim multi-task (so với 2 single-task) | outputs/models/claim | chọn theo dev; ghi cả 2 vào ablation |
| B4 | noise_filter QE + CL; ma trận transfer E3/E5 | metrics/exp3,5.json | bảng zero-shot vs translate-train × lọc |
| B5 | eval trên VN human-eval set (khi user gán xong) | metrics/vn_eval.json | F1 theo trụ, tách E vs S/G |

## 3. Phase C — Grounding (T7)

| Bước | Module | Deliverable | Done-when |
|---|---|---|---|
| C1 | evidence_pool + retriever | pool stats per doc | quy tắc pool đúng spec 03 §1 |
| C2 | nli + support, grounded@θ | outputs/grounding/claims_grounded.parquet | giữ phân phối; 3 mức θ |
| C3 | xuất 100 cặp spot-check cho user | data/annotation/grounding_check.csv | precision per θ → metrics/exp4.json |

## 4. Phase D — Chỉ số & validation (T8, T9)

| Bước | Module | Deliverable | Done-when |
|---|---|---|---|
| D1 | cti + bootstrap + min-n pooling | outputs/index/cti.parquet | CI đầy đủ; ô n<30 chỉ vào pooled |
| D2 | disclosure | cột share trong cti.parquet | khớp spec 04 §3 |
| D3 | bank_signals.csv (user gán tay, code chỉ load) + known_group | metrics/exp6.json | kết luận kể cả null |
| D4 | synthetic + sensitivity | metrics/exp7,8.json | CTI đơn điệu theo mức trộn; Kendall τ |

## 5. Phase E — Đóng gói (T10, T11, T12)

- E1: case study — trích top câu cheap-talk per bank (notebook/markdown).
- E2: đóng gói dataset translate-train (README nguồn gốc + license từng nguồn).
- E3: dọn code cũ (T12) sau khi `run.py --stage all` chạy trọn 1 bank.
- E4: bản thảo — bảng E1–E8 + hình + limitations (L1–L5, spec 00 §5).

## 6. Việc cần user làm (không code được)

1. Gán nhãn VN human-eval set (~300 câu × 3 nhãn) — Phase A4 xuất file.
2. Gán `bank_signals.csv` (10 bank × 5 năm × 5 cột) — nguồn: BCTN, HOSE/VNSI, SBV.
3. Spot-check 100 cặp grounding — Phase C3 xuất file.
4. Accept license HF cho `Unbabel/wmt22-cometkiwi-da` (nếu dùng QE chính; fallback LaBSE).
5. ~~Đăng ký/tải ML-Promise~~ ĐÃ TẢI (`data/external/ml_promise/`, kiểm kê spec 01 §7.1).
   Còn lại: **dịch sang VI 1.200 mẫu EN+FR+JA** (user hỗ trợ dịch; ZH 146 positive tùy chọn,
   KO bỏ vì không có text) → augment commitment S/G + eval grounding.
