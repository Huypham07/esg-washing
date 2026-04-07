# ESG-Washing Comprehensive Analysis Report

Generated: 2026-04-06 23:41:42
Total sentences: 32,260
Total bank-years: 50

## 1. EWRI Decomposition

Overall EWRI: **36.35** ± 2.69
Range: [31.68, 45.75]

| Component | Contribution (pts) | % of EWRI |
|-----------|-------------------|-----------|
| Indeterminate | 33.43 | 92.0% |
| Planning | 0.69 | 1.9% |
| Implemented | 2.23 | 6.1% |

## 2. Action × Evidence Interaction

**Theoretical WRS (from formula):**
| Action / ES | ES=0 | ES=0.25 | ES=0.5 | ES=0.75 | ES=1.0 |
|-------------|------|---------|--------|---------|--------|
| Implemented | 0.08 | 0.064 | 0.048 | 0.032 | 0.016 |
| Planning | 0.3 | 0.251 | 0.203 | 0.154 | 0.105 |
| Indeterminate | 0.55 | 0.481 | 0.413 | 0.344 | 0.275 |

**Empirical Distribution (actual data):**
| Action | Evidence Level | Count | % | Avg WRS |
|--------|---------------|-------|---|---------|
| Implemented | None (0) | 4075 | 12.63% | 0.080 |
| Implemented | Low (0-0.3) | 4085 | 12.66% | 0.070 |
| Implemented | Medium (0.3-0.6) | 2014 | 6.24% | 0.054 |
| Implemented | High (0.6+) | 228 | 0.71% | 0.038 |
| Planning | None (0) | 440 | 1.36% | 0.295 |
| Planning | Low (0-0.3) | 302 | 0.94% | 0.263 |
| Planning | Medium (0.3-0.6) | 157 | 0.49% | 0.220 |
| Planning | High (0.6+) | 13 | 0.04% | 0.172 |
| Indeterminate | None (0) | 10230 | 31.71% | 0.550 |
| Indeterminate | Low (0-0.3) | 6285 | 19.48% | 0.500 |
| Indeterminate | Medium (0.3-0.6) | 3956 | 12.26% | 0.433 |
| Indeterminate | High (0.6+) | 475 | 1.47% | 0.377 |

## 3. Topic Analysis

| Topic | N | % | EWRI | Risk | Impl% | Indet% | EvRate% |
|-------|---|---|------|------|-------|--------|---------|
| E | 3029 | 9.4% | 33.3 | Medium | 37.6% | 57.2% | 30.1% |
| S_labor | 5584 | 17.3% | 36.4 | Medium | 33.1% | 65.2% | 21.1% |
| S_community | 1996 | 6.2% | 32.9 | Medium | 41.2% | 58.1% | 27.0% |
| S_product | 4866 | 15.1% | 34.9 | Medium | 36.6% | 60.4% | 23.6% |
| G | 16785 | 52.0% | 37.1 | Medium | 28.7% | 68.4% | 24.8% |

## 4. Temporal Trends

| Year | EWRI Mean | Std | Old EWRI | Ev Rate | Impl% | Indet% |
|------|-----------|-----|----------|---------|-------|--------|
| 2020 | 37.08 | 4.13 | 54.92 | 0.236 | 0.308 | 0.665 |
| 2021 | 35.67 | 2.92 | 53.59 | 0.269 | 0.340 | 0.639 |
| 2022 | 35.98 | 1.79 | 53.94 | 0.261 | 0.326 | 0.650 |
| 2023 | 36.13 | 2.13 | 54.18 | 0.241 | 0.321 | 0.656 |
| 2024 | 36.90 | 2.08 | 54.95 | 0.240 | 0.302 | 0.666 |

## 5. Cross-Bank Comparison

| Rank | Bank | Avg EWRI | Risk | Strengths | Weaknesses |
|------|------|----------|------|-----------|------------|
| 1 | agribank | 34.10 | Medium | Lower-than-average washing risk; Better evidence coverage | - |
| 2 | vpbank | 35.05 | Medium | - | - |
| 3 | mbbank | 35.07 | Medium | Better evidence coverage | - |
| 4 | bidv | 35.17 | Medium | - | - |
| 5 | ocb | 35.34 | Medium | - | - |
| 6 | viettinbank | 35.34 | Medium | - | - |
| 7 | shb | 36.03 | Medium | - | - |
| 8 | techcombank | 37.18 | Medium | - | High proportion of vague claims (>65%) |
| 9 | vietcombank | 38.72 | High | - | Higher-than-average washing risk; High proportion of vague claims (>65%) |
| 10 | bsc | 41.50 | High | - | Higher-than-average washing risk; Weak evidence coverage |

## 6. Correlation Analysis

**EWRI Correlations (Spearman):**
| Feature | ρ (vs EWRI) | Interpretation |
|---------|-------------|----------------|
| ewri_old | 0.998 | strong positive |
| indeterminate_ratio | 0.983 | strong positive |
| implemented_ratio | -0.977 | strong negative |
| evidence_ratio | -0.672 | moderate negative |
| topic_entropy | -0.379 | weak negative |
| planning_ratio | -0.180 | weak negative |
| avg_evidence_strength | -0.098 | weak negative |

**Action Label Effect Size**: η² = 0.9517 (large effect)

## 7. Evidence Type Importance

| Type | Freq% | Avg WRS (present) | Avg WRS (absent) | Risk Reduction |
|------|-------|------------------|-----------------|----------------|
| KPI | 6.92% | 0.092 | 0.381 | 75.8% |
| Time_bound | 11.35% | 0.195 | 0.382 | 48.9% |
| Standard | 5.69% | 0.273 | 0.366 | 25.3% |
| Third_party | 5.15% | 0.298 | 0.364 | 18.3% |

## 8. Formula Comparison (Old Additive vs New Interaction)

| Metric | Old Formula | New Formula | Change |
|--------|------------|------------|--------|
| mean | 54.32 | 36.35 | -17.97 |
| std | 2.85 | 2.69 | -0.16 |
| iqr | 3.37 | 2.88 | -0.49 |
| range | 13.01 | 14.07 | +1.06 |
| cv | 5.3 | 7.4 | +2.10 |

Rank Correlation: 0.9975
Std Improvement: -5.6%
Range Improvement: +8.1%

## 9. Quality Verification

Top risk claims noise rate: **0.0%** (0/100)

### High Risk (Pure Washing)

1. **[agribank 2023]** [Indeterminate] WRS=0.715
   > "chính sách và 02 chương trình mục tiêu quốc gia, xây dựng các sản phẩm tín dụng ưu đãi phù hợp với nhiều đối tượng khách hàng, phát triển đa dạng các ..."

2. **[bidv 2020]** [Indeterminate] WRS=0.715
   > "GIA TĂNG NỀN KHÁCH HÀNG THEO HƯỚNG BỀN VỮNG, PHÙ HỢP MỤC TIÊU VÀ ĐỊNH HƯỚNG CHIẾN LƯỢC CỦA BIDV..."

3. **[bidv 2020]** [Indeterminate] WRS=0.715
   > "- Đầu tư mua sắm, tự phát triển các ứng dụng, kênh phân phối hiện đại, cung cấp các sản phẩm dịch vụ đa dạng, mang đến trải nghiệm hiện đại, tiên tiến..."

### Suspicious (Implemented without Evidence)

1. **[agribank 2022]** [Implemented] WRS=0.104
   > "000 căn nhà tình nghĩa, nhà đại đoàn kết cho người nghèo và đối tượng chính sách trong cả nước, 17 công trình trường học và phòng học, 06 trạm và cơ s..."

2. **[agribank 2024]** [Implemented] WRS=0.104
   > "Kết quả đến 31/12/2024, doanh số cho vay lũy kế từ đầu chương trình đạt gần 90...."

3. **[bidv 2020]** [Implemented] WRS=0.104
   > "- Tổ chức cuộc thi online "Nét đẹp văn hóa BIDV", đã thu hút được gần 40...."

### Low Risk (Substantive)

1. **[bsc 2021]** [Implemented] WRS=0.028
   > "Nhiều sản phẩm mới được triển khai, nâng cấp trong năm 2021 như phần mềm xác nhận lệnh online, tính năng mở tài khoản trực tuyến trên website của BSC ..."
   Evidence types: Third_party, Time_bound

2. **[vpbank 2024]** [Implemented] WRS=0.030
   > "VPBank và Ngân hàng Hợp tác Quốc tế Nhật Bản (JBIC) ký kết hợp đồng tín dụng trị giá lên tới 150 triệu USD nhằm tài trợ cho các dự án năng lượng tái t..."
   Evidence types: KPI, Standard, Time_bound

3. **[shb 2021]** [Implemented] WRS=0.031
   > "SHB đã hoàn tất 03 trụ cột của Hiệp ước vốn Basel II từ năm 2020 và tiếp tục triển khai Basel II theo phương pháp nâng cao, đáp ứng tuân thủ toàn diện..."
   Evidence types: Standard, Time_bound

---
*Analysis generated by ESG-Washing Detection Framework*