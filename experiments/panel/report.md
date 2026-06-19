# Panel ESG-washing — 9 ngan hang x 5 nam (45 panel)

## RQ1 — Prevalence (CTI/NAR/QDR)
- CTI = 0.372 (sd 0.0919, CI95 [0.3452, 0.3995]) · NAR = 0.3094 · QDR = 0.3186

**Theo nam**

| year | cti | nar | qdr | n_commit |
| --- | --- | --- | --- | --- |
| 2020 | 0.3781 | 0.2974 | 0.3245 | 43.1111 |
| 2021 | 0.3352 | 0.2909 | 0.3738 | 49.2222 |
| 2022 | 0.4013 | 0.3261 | 0.2726 | 65.2222 |
| 2023 | 0.3811 | 0.3206 | 0.2983 | 80.0 |
| 2024 | 0.3644 | 0.3119 | 0.3237 | 101.2222 |


**Theo ngan hang**

| bank | cti | nar | qdr | n_commit |
| --- | --- | --- | --- | --- |
| agribank | 0.2927 | 0.3489 | 0.3584 | 48.2 |
| bidv | 0.373 | 0.326 | 0.301 | 89.0 |
| mbbank | 0.4598 | 0.2452 | 0.2951 | 85.6 |
| ocb | 0.296 | 0.4444 | 0.2596 | 42.0 |
| shb | 0.4157 | 0.2828 | 0.3015 | 86.0 |
| techcombank | 0.397 | 0.2751 | 0.3279 | 76.6 |
| vietcombank | 0.2891 | 0.2664 | 0.4445 | 30.2 |
| viettinbank | 0.4306 | 0.3258 | 0.2436 | 82.4 |
| vpbank | 0.3942 | 0.27 | 0.3358 | 69.8 |


## RQ2 — Selective disclosure
- Share trung binh: E=0.241 · S=0.3486 · G=0.4104
- Friedman chi2=50.0678, p=1.343e-11 (n_block=45)
- Wilcoxon env_vs_gov: p=3.26e-09, median_diff=-0.1613

## RQ3 — Substance gap / xu huong
- CTI~year Spearman rho=0.0309 (p=0.8405)
- QDR~year Spearman rho=-0.0732 (p=0.6327)
- n_commit~year rho=0.6087 (p=9.135e-06)

## Validation — on dinh ranking (sensitivity)
- Bootstrap Kendall tau mean=0.7284 (p05=0.5), top-1 retention=0.6725
- Leave-one-year-out tau_min=0.6667