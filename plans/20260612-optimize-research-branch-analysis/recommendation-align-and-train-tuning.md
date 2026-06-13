# Đề xuất (provisional — CHƯA CHỐT): align data + giữ kiến trúc + hybrid train/tuning

> Brainstorm-technical 2026-06-13. Bối cảnh: 2 track song song; user giữ kiến trúc 5-binary của mình,
> align data sang đồng nghiệp, hỏi train/tuning hướng nào tốt hơn. User đã trả lời: compute rộng rãi,
> **bỏ hẳn silver**, **re-tune toàn bộ**. Đây là ĐỀ XUẤT, user chưa chốt thực thi.

## 1. Vấn đề & yêu cầu
- Track user: 5 model nhị phân rời (PhoBERT-v2, AutoModelForSequenceClassification, weighted-CE), 2 nguồn nhãn
  (translate-train `vi_gold` + silver LLM `vi_silver`), downstream CTI ĐÃ chạy. Yếu: gov/commit/spec F1 .73-.74.
- Mục tiêu: nâng rigor lên mức bảo vệ được trước hội đồng, **giữ kiến trúc hiện tại**, học cái tốt của đồng nghiệp.

## 2. Ba quyết định

### D1 — Data/labels: ALIGN sang đồng nghiệp (translate-train + cross-label, BỎ silver)
- **Nguồn nhãn DUY NHẤT = translate-train** (`vi_gold`, đã có sẵn). Bỏ hoàn toàn silver (`vi_silver`, `labels_silver`,
  `silver_*_labeler.py`, model `*_silver`).
- **Thêm ESGBERT cross-label** (Schimanski 2023): điền ô NaN topic. *Adapter cho 5-binary:* mỗi model train trên
  subset cột non-NaN của nó (vd env model = mọi dòng env≠NaN, gồm dòng cross-label từ tập soc/gov). Không cần đổi arch.
- **Thêm augment S/G** (`action_500` + `ml_promise`) → CHỈ vào commitment model. **Specificity KHÔNG có augment**
  (augment thiếu nhãn specificity) → specificity vẫn climate-only → khai báo limitation.

### D2 — Kiến trúc: GIỮ NGUYÊN 5 binary (user chốt)
- AutoModelForSequenceClassification × 5, phobert-base-v2, weighted-CE. Defensible; chỉ cần data-adapter ở D1.
- Trả giá: 5× chi phí tune/seed so với 2 model multi-head — chấp nhận được vì compute rộng rãi.

### D3 — Train/tuning: HYBRID (giữ 3, lấy 4) + re-tune toàn bộ
**GIỮ của user:** HF Trainer engine · weighted-CE loss · EarlyStoppingCallback.
**LẤY từ đồng nghiệp:**
1. **Threshold tuning per-model** trên val (tối đa F1) thay argmax 0.5 — ngưỡng tối ưu thường <0.5 cho data lệch.
2. **Multi-seed 5** [42-46] → báo cáo **mean±std** (Dodge 2020/Mosbach 2021: fine-tune BERT data nhỏ bất ổn theo seed).
3. **Search space mới:** BỎ `epochs` (đã có early-stopping), THÊM `warmup_ratio` + `dropout` (giữ lr/batch/wd).
4. **Sửa pruner:** thêm Optuna `TrainerCallback` report dev-macro-F1 mỗi epoch → MedianPruner prune thật
   (hiện pruner cấu hình nhưng không report → vô dụng).
**KHÔNG lấy:** masked BCE (chỉ cho multi-head, không hợp 5-binary).
**Protocol:** tune trên 1 seed cố định (rẻ) → train cuối 5 seed với best-params. Không multi-seed khi tuning.

## 3. Phụ thuộc chéo (PHÁT SINH — phải xử lý cùng)

🔴 **VN human-eval set thành P0 BẮT BUỘC** (hệ quả của bỏ silver): silver có lợi thế DUY NHẤT là *in-domain*
(train trên câu bank VN thật). Bỏ silver = chỉ còn translate-train *out-domain* (climate EN dịch) → **không có gì
xác minh model chạy đúng trên corpus bank VN thật** cho tới khi có human-eval 300 câu. Trước đây human-eval là
"nên có"; sau quyết định bỏ silver, nó là "không có thì không kết luận được".

🔴 **Threshold tuning ⟹ CTI sensitivity analysis** (coupling): ngưỡng commit/spec ảnh hưởng trực tiếp CTI.
Tune ngưỡng mà không quét sensitivity (Kendall τ ổn định ranking) = p-hacking chỉ số dưới mắt hội đồng.

🟠 **Re-run downstream CTI trên model gold-track mới** — số CTI demo cũ (E.52/S.54/G.60) chạy trên silver,
sẽ đổi. Code CTI giữ nguyên, chỉ chạy lại.

## 4. Pros / Cons

**Pros:** rigor tăng mạnh (multi-seed mean±std, threshold tuned, pruning thật, nguồn nhãn defensible);
giữ kiến trúc + downstream đã chạy của user (không phá việc đã làm); vá gov/commit (augment + cross-label).

**Cons / rủi ro:** (a) bỏ silver = mất tín hiệu in-domain → phụ thuộc human-eval; (b) specificity vẫn yếu
(augment không chữa) → cần khai báo trần α=0.17; (c) 5 model × 5 seed × re-tune = nhiều compute (đã chấp nhận);
(d) translate-train domain-shift climate→S/G chưa rõ mức cho tới human-eval.

## 5. Tiêu chí thành công
- gov/commitment F1 cải thiện rõ so với .73-.74 hiện tại (đo trên gold test).
- mean±std 5 seed báo cáo đủ; pruner cắt được trial tệ (log số trial pruned).
- VN human-eval: F1 theo trụ, tách E vs S/G; κ inter-annotator.
- CTI ranking ổn định qua sensitivity (Kendall τ) khi quét ngưỡng ±0.05 và θ.

## 6. Next steps (thứ tự logic — chưa phải plan chính thức)
1. Data: adapter cross-label → 5 per-task set; thêm augment commitment; gỡ silver.
2. Tuning: search space mới + pruner fix; re-tune 5 model (1 seed).
3. Train: 5-seed final + threshold tuning per-model.
4. Eval: xuất + gán VN human-eval set (việc người, song song).
5. Downstream: chạy lại CTI trên model mới + thêm sensitivity analysis.

## 7. Câu hỏi chưa giải quyết
1. F1 .94/.90 trước đây là trên gold (translate) hay silver test? (xác nhận baseline để đo cải thiện).
2. ESGBERT cross-label: chấp nhận tải 3 model HF ESGBERT (public, không gated) + chạy inference 1 lần? GPU đủ.
3. `ml_promise_vi.csv` đã dịch xong chưa, hay user phải dịch (1200 mẫu EN+FR+JA)?
4. Bỏ silver có cần giữ lại bản backup branch không (phòng khi muốn dựng ablation in-domain sau)?
