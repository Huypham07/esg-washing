# Phase 04 — VN human-eval set (P0, human-gated, SONG SONG)

**Context:** recommendation §3 (phụ thuộc chéo) · tham chiếu `git show origin/optimize-research:src/esgwash/data/vn_eval_set.py`
**Priority:** 🔴 P0 BẮT BUỘC · **Status:** pending · **Depends:** 01 (corpus sạch); eval cần 03

## Overview
Test set THẬT trên tiếng Việt (in-domain bank) — bắt buộc vì bỏ silver mất tín hiệu in-domain, translate-train out-domain chưa verify. Sample 300 câu stratified → user gán → đo F1 thật + Cohen κ. Khởi động SỚM (gán tay tốn thời gian).

## Key insights
- Gold EN test chỉ là upper-bound; human-eval VN là test CHÍNH để claim transfer.
- Tách eval theo trụ E vs S/G → định lượng domain shift climate→S/G (thay vì thừa nhận suông).
- specificity κ dự kiến thấp (α≈0.17 gốc) → khai báo, không ép.

## Requirements
- ~300 câu corpus stratified theo bank × trụ-dự-đoán (dùng preds Phase 03).
- Gán 5 nhãn/câu: env,soc,gov,commitment,specificity. (Tùy chọn) annotator thứ 2 → κ.

## Related Code Files
- **Create:** `src/training/data_prep/vn_eval_set.py` (port từ esgwash) — `sample_for_annotation`, `load_annotations`, `inter_annotator_kappa`.
- **Create:** `src/training/eval/eval_vn_human.py` — load model + threshold (Phase 02/03) → eval trên human-eval, F1 per task, tách E vs S/G.
- Output: `data/annotation/vn_eval_todo.csv` (xuất), `data/annotation/vn_eval_done.csv` (user điền), `outputs/metrics/vn_eval.json`.

## Implementation Steps
1. Port `vn_eval_set.py`: sample 300 stratified bank×pillar (preds topic Phase 03); xuất CSV cột sent_id/doc_id/bank/year/sentence/ctx + 5 cột nhãn rỗng.
2. **User gán** 300 câu (việc người — bắt đầu NGAY khi có CSV, song song Phase 02/03). Tùy chọn: người thứ 2 gán mẫu con → κ.
3. `eval_vn_human.py`: load 5 model + threshold → predict trên câu đã gán → F1 per task; bảng tách E vs S/G; κ nếu có annotator 2.
4. Ghi `vn_eval.json` + nhận xét: transfer tốt không? S/G tụt bao nhiêu so với E?

## Todo
- [ ] Port `vn_eval_set.py` + xuất CSV 300 câu
- [ ] (USER) gán nhãn 300 câu — SONG SONG
- [ ] (tùy chọn) annotator 2 + κ
- [ ] `eval_vn_human.py` → F1 thật per task, tách E/S/G
- [ ] Ghi vn_eval.json + nhận xét transfer

## Success Criteria
- Bộ 300 câu VN gán xong; `vn_eval.json` có F1 per task trên tiếng Việt thật.
- Có số tách E vs S/G (định lượng domain shift).
- Nếu κ đo được: báo cáo per nhãn (đặc biệt specificity thấp = kỳ vọng).

## Risks
- **Effort gán tay** = nút thắt thời gian → bắt đầu sớm; có thể giảm còn 200 nếu gấp (nhưng compute rộng rãi ≠ thời gian người).
- Transfer S/G có thể kém (climate→bank S/G) → giới hạn claim paper về S/G, không giấu.
- 1 annotator → không có κ → nêu rõ là hạn chế (self-annotation).
