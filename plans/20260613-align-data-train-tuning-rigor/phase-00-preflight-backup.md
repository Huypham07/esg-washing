# Phase 00 — Pre-flight (local, KHÔNG git)

**Context:** `plans/20260612-optimize-research-branch-analysis/recommendation-align-and-train-tuning.md`
**Priority:** P0 · **Status:** pending · **Effort:** ~10ph

## Overview
User chốt: **KHÔNG commit, làm trực tiếp trên local.** Phase này chỉ dựng lưới an toàn nhẹ (copy file thuần, không git) + neo baseline + nhắc KHÔNG pull nhánh đồng nghiệp. Silver KHÔNG xóa.

## Requirements
- Không dùng git (không branch/commit/stash). Mọi an toàn = copy file trên đĩa.
- Phục hồi được nếu Phase 01/02 lỡ ghi đè/hỏng file đang dùng.

## Key insights
- Rủi ro của "no-git": sửa hỏng `train_model.py`/`prepare_gold_splits.py` → KHÔNG có git undo cho phần uncommitted. → giảm nhẹ bằng copy thư mục trước khi sửa.
- `data/vi_gold/*` sẽ bị Phase 01 GHI ĐÈ (cross-label + augment). Bản gốc **tái tạo được** bằng `prepare_gold_splits` (deterministic seed=42) nhưng copy 1 lần cho nhanh/an toàn.
- Silver để nguyên trên đĩa = không mất gì; chỉ ngừng tham chiếu trong Phase 01.

## Implementation Steps
1. Copy nhẹ (thuần file, không git) các thứ Phase 01/02 sẽ đụng:
   - `data/vi_gold/` → `data/_local_backup/vi_gold_pre_crosslabel/` (Phase 01 ghi đè).
   - `src/training/train_model.py`, `tune_hyperparams.py`, `prepare_gold_splits.py`, `config/train.yml` → `src/_local_backup/` (Phase 02 sửa). (Tùy chọn — nếu tự tin có thể bỏ.)
2. KHÔNG `git pull`/merge `origin/optimize-research`. Tham chiếu `src/esgwash` qua `git show origin/optimize-research:<path>` (chỉ đọc, không đụng working tree).
3. Xác nhận baseline gold đã ghi (outputs/compare/silver_vs_gold_summary.json) — bảng trong plan.md.
4. Ghi rõ: **silver KHÔNG xóa** ở Phase 01 (chỉ ngừng dùng).

## Related Code Files
- Create: `data/_local_backup/` (copy, không git). Modify: none.

## Todo
- [ ] Copy `data/vi_gold/` → `data/_local_backup/vi_gold_pre_crosslabel/`
- [ ] (tùy chọn) Copy 4 file code/config Phase 02 sẽ sửa → `src/_local_backup/`
- [ ] KHÔNG pull nhánh đồng nghiệp (chỉ `git show` khi cần đọc)
- [ ] Xác nhận baseline numbers ghi trong plan

## Success Criteria
- Có bản copy `vi_gold` gốc (khôi phục được sau khi Phase 01 ghi đè).
- Không thực hiện thao tác git nào làm thay đổi branch/working tree.

## Risks
- No-git = không undo được cho code uncommitted nếu sửa hỏng → copy thư mục `src/_local_backup/` là lưới chính.
- Quên: `data/_local_backup/` nên thêm vào `.gitignore` để khỏi lỡ commit sau này (không bắt buộc giờ).
