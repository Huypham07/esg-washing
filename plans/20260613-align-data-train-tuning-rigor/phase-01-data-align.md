# Phase 01 — Data align (rút gọn): action_500 augment + ngừng dùng silver

**Context:** recommendation D1 · **đã ĐO và quyết BỎ cross-label** (`outputs/_measure_crosslabel.py`)
**Priority:** P0 · **Status:** pending · **Depends:** 00

## Quyết định: BỎ ESGBERT cross-label (data-backed 2026-06-13)
Đo thực tế ESGBERT trên ô NaN: chỉ **5% fill là positive cho gov** (+82 pos / +1465 neg), balance gov **xấu đi 0.27→0.17**. Với kiến trúc **5 binary rời** (khác multi-head của đồng nghiệp), cross-label chủ yếu thêm negative, +pos khiêm tốn → **lợi ích marginal, BỎ.** Lever thật cho gov = threshold-tuning + multi-seed (Phase 02/03).
→ **Topic env/soc/gov GIỮ NGUYÊN** `data/vi_gold/*` (translate-train từ `prepare_gold_splits`) → **baseline cũ (env.955/soc.916/gov.821) còn nguyên để so** (val/test KHÔNG đổi).

## Overview
Phase 01 thu hẹp còn 2 việc: (1) augment commitment bằng action_500; (2) ngừng dùng silver (trỏ gold). **KHÔNG** cross-label, **KHÔNG** module mới, **KHÔNG** tải ESGBERT, **KHÔNG** đổi split topic.

## Requirements
- commitment train += action_500 positives; specificity bất biến; topic bất biến.
- Đường chạy chính (train→classify→CTI) dùng gold/translate-train, không silver.

## Related Code Files
- **Modify:** `src/training/data_prep/prepare_gold_splits.py` — nhánh commitment: train += `data/en_gold/translate/action_500.csv` (action→commitment). Topic giữ nguyên.
- **NGỪNG DÙNG (KHÔNG xóa):** silver — đảm bảo `config/train.yml` (đã trỏ `data/vi_gold`) + `src/pipeline/run_cti.py`/`classify_corpus.py` mặc định track gold. KHÔNG đụng file silver.
- **Ghi đè** (đã backup Phase 00): `data/vi_gold/commitment/train.parquet`. Topic/specificity bất biến.

## Implementation Steps
1. ✅ `prepare_gold_splits.py` nhánh commitment: đọc `action_500.csv` (translate); map `action→commitment` **CẢ 500** (quyết định 2026-06-13: thêm cả 2 nhãn để GIỮ balance 0.42 — action_500 vốn 43% pos; "chỉ positive" sẽ lệch lên 0.54 = méo prior → CTI). Train-only (phá alignment commit↔spec sau augment — CHẤP NHẬN vì train binary rời). Leak-guard câu trùng commitment test (=0).
2. Xác nhận đường chạy chính trỏ gold (config + pipeline). KHÔNG xóa silver.
3. Ghi TODO ml_promise (deferred — không có trong data user; cần lấy nhánh đồng nghiệp + dịch 1200 mẫu nếu sau này muốn vá thêm S/G commitment).

## Kết quả (✅ done 2026-06-13)
- commitment train **841 → 1341** (+500 action_500), pos-rate **0.42 giữ nguyên**; commit train↔test leak **0**.
- specificity **841 bất biến**; topic env/soc/gov **hash-identical với backup** (baseline còn nguyên để so).
- config/train.yml đã trỏ `data/vi_gold` (training dùng gold). Silver giữ trên đĩa (không xóa).
- **Fix bug câu rỗng (2026-06-13):** `_std` thêm lọc empty sau NFC → bỏ 6 câu rỗng/test ở commit & spec (có cả nhãn=1 cho câu rỗng = rác). commit/spec **test 320→314**; topic & train/val không đổi (topic hash-identical → tuning topic an toàn). ⚠️ baseline cũ commit/spec (.795/.792) đo trên test 320-có-rác → số mới trên 314-sạch (chênh nhỏ, test mới hợp lệ hơn).

## Todo
- [x] commitment train += action_500 (CẢ 500, balance 0.42 giữ nguyên)
- [x] Xác nhận config trỏ gold (config/train.yml → data/vi_gold; silver còn trên đĩa)
- [x] Ghi TODO ml_promise (deferred — không có trong data user)
- [x] đo + quyết bỏ cross-label — `outputs/_measure_crosslabel.py`

## Success Criteria
- `data/vi_gold/commitment/train.parquet` có thêm rows action_500; pos_rate hợp lý (log).
- Topic vi_gold KHÔNG đổi (baseline còn nguyên để so).
- Đường chạy chính không dùng silver; file silver vẫn còn trên đĩa.

## Risks
- action=0→commitment=0 có thể nhiễu (giống bài học cross-label) → ưu tiên thêm positives; nếu thêm cả negatives thì coi như ablation, đo lại.
- commitment imbalance đổi sau augment → weighted-CE đã bù; theo dõi pos_rate.
