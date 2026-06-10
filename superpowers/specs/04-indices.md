# Spec 04 — Chỉ số washing

Nguyên tắc: chỉ số = **tỉ lệ giữa output các classifier**, không trọng số tự đặt.
Đơn vị phân tích: ô (bank `b`, year `y`, pillar `p` ∈ {E, S, G}).

## 1. CTI (Cheap Talk Index — Bingler 2022)

```
C(b,y,p)   = {câu: commitment=1, pillar=p}
CTI(b,y,p) = |{c ∈ C : specific=0}| / |C|
```

CTI ∈ [0,1]; cao = nhiều cam kết suông.

## 2. Grounded-CTI (đóng góp chính)

```
grounded-CTI(b,y,p) = |{c ∈ C : specific=0  HOẶC  (specific=1 VÀ support<θ)}| / |C|
```

- Báo cáo tách bạch CTI và grounded-CTI (chênh lệch = "cheap talk ẩn").
- θ ∈ {0.5, 0.7, 0.9}; bảng chính dùng 0.7, phụ lục đủ 3 mức (spec 03 §4).

## 3. Selective disclosure (cherry-picking)

- Phân bố tỉ trọng câu ESG theo trụ: `share(p) = n_p / (n_E+n_S+n_G)` per (bank, year).
- Đo độ lệch so với phân bố trung bình toàn ngành cùng năm (chênh lệch share, kèm CI).
- Diễn giải: lệch mạnh khỏi mặt bằng ngành = dấu hiệu né trụ khó (Bingler 2021; Rouen 2023) —
  trình bày descriptive, không gộp vào CTI.

## 4. Uncertainty & ngưỡng tin cậy

- **Bootstrap CI:** resample câu commitment trong ô (B=1000, percentile 95%).
- **Ngưỡng n tối thiểu:** ô có |C| < 30 → không xếp hạng riêng, chỉ hiện trong bảng
  pooled (gộp các năm: bank × pillar, hoặc gộp trụ: bank × year).
- Cảnh báo trước (đã thấy ở pipeline cũ): số commitment per ô ở S/G có thể nhỏ —
  pooled ranking là kết quả chính, per-year là phụ.

## 5. Output

- `outputs/index/cti.parquet`: `bank, year, pillar, n_commit, cti, cti_lo, cti_hi,
  gcti@θ, gcti_lo, gcti_hi, share_pillar`
- `outputs/index/ranking.md`: bảng xếp hạng pooled + per-year (chỉ ô đủ n).
