# Panel ESG-washing — 9 ngan hang x 5 nam (45 panel)

## RQ1 — Prevalence (CTI/NAR/QDR)
- CTI = 0.3833 (sd 0.0832, CI95 [0.3601, 0.4077]) · NAR = 0.313 · QDR = 0.3037

**Theo nam**

| year | cti | nar | qdr | n_commit |
| --- | --- | --- | --- | --- |
| 2020 | 0.4019 | 0.2987 | 0.2994 | 48.2222 |
| 2021 | 0.3635 | 0.3104 | 0.3261 | 52.7778 |
| 2022 | 0.4013 | 0.3261 | 0.2726 | 65.2222 |
| 2023 | 0.3811 | 0.3206 | 0.2983 | 80.0 |
| 2024 | 0.3688 | 0.309 | 0.3223 | 107.5556 |


**Theo ngan hang**

| bank | cti | nar | qdr | n_commit |
| --- | --- | --- | --- | --- |
| agribank | 0.2927 | 0.3489 | 0.3584 | 48.2 |
| bidv | 0.373 | 0.326 | 0.301 | 89.0 |
| mbbank | 0.4598 | 0.2452 | 0.2951 | 85.6 |
| ocb | 0.296 | 0.4444 | 0.2596 | 42.0 |
| shb | 0.4157 | 0.2828 | 0.3015 | 86.0 |
| techcombank | 0.397 | 0.2751 | 0.3279 | 76.6 |
| vietcombank | 0.3907 | 0.2987 | 0.3106 | 57.2 |
| viettinbank | 0.4306 | 0.3258 | 0.2436 | 82.4 |
| vpbank | 0.3942 | 0.27 | 0.3358 | 69.8 |


## RQ2 — Selective disclosure
- Share trung binh: E=0.2401 · S=0.357 · G=0.4029
- Friedman chi2=53.625, p=2.267e-12 (n_block=45)
- Wilcoxon env_vs_gov: p=2.027e-09, median_diff=-0.1596

## RQ3 — Substance gap / xu huong
- CTI~year Spearman rho=-0.0865 (p=0.572)
- QDR~year Spearman rho=0.0678 (p=0.6583)
- n_commit~year rho=0.6675 (p=5.496e-07)

## Validation — on dinh ranking (sensitivity)
- Bootstrap Kendall tau mean=0.7192 (p05=0.5), top-1 retention=0.6265
- Leave-one-year-out tau_min=0.7222