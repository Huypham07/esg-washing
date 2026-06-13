---
status: pending
created: 2026-06-13
owner: dqd
relates: plans/20260612-optimize-research-branch-analysis/recommendation-align-and-train-tuning.md
---

# Plan: Align data + nâng rigor train/tuning (giữ kiến trúc 5-binary)

> Nâng track của user (`src/training`+`src/pipeline`) lên mức bảo vệ-được-hội-đồng bằng cách
> mượn cái tốt của đồng nghiệp (`src/esgwash`) NHƯNG GIỮ kiến trúc 5 model nhị phân rời.
> Nguồn sự thật: `plans/20260612-optimize-research-branch-analysis/recommendation-align-and-train-tuning.md`.

## Mục tiêu & baseline (gold/translate-train TEST macro-F1 — số phải vượt)

| Task | Baseline gold | f1_pos | tfidf_lr | Ghi chú |
|---|---|---|---|---|
| env | 0.955 | 0.940 | 0.865 | mạnh |
| soc | 0.916 | 0.899 | 0.844 | mạnh |
| gov | 0.821 | 0.733 | 0.759 | **yếu** — đích cải thiện |
| commitment | 0.795 | 0.726 | 0.734 | **yếu** — đích cải thiện (augment) |
| specificity | 0.792 | 0.737 | 0.732 | **yếu** — augment KHÔNG chữa (khai báo limitation) |

🔴 **Kỳ vọng thực tế:** silver từng cho gov/commit/spec cao hơn (.85-.89) nhưng **circular** (test=nhãn LLM).
Bỏ silver ⇒ headline 3 task này = số gold honest. Lever kéo lại: **commitment←action_500 augment** (chắc);
**gov/spec←threshold-tuning + multi-seed** (Phase 02/03). **Cross-label đã BỎ** (đo ra marginal cho kiến trúc
binary: gov chỉ +82 pos / +1465 neg, balance xấu → `outputs/_measure_crosslabel.py`). Baseline topic GIỮ NGUYÊN (so sạch).

## Phases

| # | Phase | Status | Phụ thuộc | Tóm tắt |
|---|---|---|---|---|
| 00 | [Pre-flight (local, no-git)](phase-00-preflight-backup.md) | ✅ done | — | KHÔNG commit; copy vi_gold(15/15)+4 code file → _local_backup; baseline OK |
| 01 | [Data align (rút gọn): augment + ngừng silver](phase-01-data-align.md) | ✅ done | 00 | commit train 841→1341 (+action_500, bal 0.42); topic/spec bất biến; cross-label BỎ |
| 02 | [Train/tuning hybrid upgrade](phase-02-train-tuning-upgrade.md) | ✅ done | 00 | **ngưỡng 0.5 (cách C, bỏ tự-chọn)** + 5-seed + warmup + pruner; smoke PASS; reviewed |
| 03 | [Re-tune + re-train 5 models](phase-03-retune-retrain.md) | ✅ done | 01,02 | env.954/soc.911/spec.808/gov.801/commit.789 (5-seed mean±std @0.5); [results](results-train-classifiers.md) |
| 04 | [VN human-eval set (P0)](phase-04-vn-human-eval.md) | pending | 01 (song song) | sample 300 câu → user gán → eval model trên bank VN thật |
| 05 | [CTI thuần re-run + sensitivity](phase-05-cti-rerun-sensitivity.md) | ✅ done (SƠ BỘ) | 03 | CTI pooled gold (27 ô, loại bsc): **E .77 / S .81 / G .94** (tương đối); G nhiễu domain (sensitivity τ .67 thấp nhất); E đáng tin nhất (τ→1.0). Đánh giá: [cti-evaluation-and-framing.md](cti-evaluation-and-framing.md) |
| 06 | Index validation (synthetic [+known-group?]) | 🟡 pending (bàn đồng nghiệp) | 05 | synthetic monotonic (code-được, **tối thiểu**) + known-group `bank_signals` (cần gán tay) — **điều kiện CTI có nghĩa khoa học** |
| 07 | grounded-CTI (mở rộng) | 🔵 future | 06 | lớp grounding (retrieval bkai + NLI mDeBERTa) → grounded-CTI; **báo cáo CÙNG CTI thuần** (chênh = cheap-talk ẩn) |

## Key dependencies / ràng buộc
- **KHÔNG commit, làm trực tiếp trên local** (quyết định user 2026-06-13). KHÔNG git branch/commit; KHÔNG pull nhánh đồng nghiệp. An toàn = copy file thuần (không git). **Silver KHÔNG xóa** — để nguyên trên đĩa, chỉ ngừng dùng (trỏ config/pipeline sang gold). Dọn sau (khi nào commit thì tính).
- **Kiến trúc GIỮ NGUYÊN** (5 binary AutoModelForSequenceClassification, phobert-base-v2). KHÔNG multi-head.
- **ml_promise augment = DEFERRED** (không có trong data; chỉ `action_500` sẵn sàng). Phase 01 optional sub-task.
- **VN human-eval = P0 bắt buộc** (hệ quả bỏ silver: mất tín hiệu in-domain). Khởi động sớm, song song.
- **threshold-tuning ⟹ CTI sensitivity** (coupling, Phase 05) — tránh p-hacking chỉ số.
- Code-style repo: snake_case .py, comment tiếng Việt, config YAML anchor, dual-use `main()`, parquet IO,
  SEED=42, `print` không `logging`. Sửa trực tiếp file hiện có, KHÔNG tạo file "enhanced".

## Câu hỏi đã giải quyết (preflight)
1. ✅ F1 .94/.90 vs .73-.74 = **gold (translate-train) test** (outputs/compare/silver_vs_gold_summary.json).
2. ✅ ml_promise **KHÔNG có** trong data user → defer; action_500 translated **có sẵn**.
3. ✅ KHÔNG commit/branch (user chốt làm local). Silver KHÔNG xóa, chỉ ngừng dùng. An toàn = copy file thuần.
