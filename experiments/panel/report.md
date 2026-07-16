# Panel ESG-washing — 9 ngan hang x 5 nam (45 panel)

## RQ1 — Prevalence (CTI/NAR/QDR)
- CTI = 0.4791 (sd 0.0923, CI95 [0.4529, 0.506]) · NAR = 0.3035 · QDR = 0.2174

**Theo nam**

| year | cti | nar | qdr | n_commit |
| --- | --- | --- | --- | --- |
| 2020 | 0.5212 | 0.2671 | 0.2116 | 48.2222 |
| 2021 | 0.4887 | 0.2994 | 0.2119 | 52.7778 |
| 2022 | 0.4936 | 0.2762 | 0.2302 | 65.2222 |
| 2023 | 0.4504 | 0.3296 | 0.22 | 80.0 |
| 2024 | 0.4417 | 0.3451 | 0.2132 | 107.5556 |


**Theo ngan hang**

| bank | cti | nar | qdr | n_commit |
| --- | --- | --- | --- | --- |
| agribank | 0.3527 | 0.3637 | 0.2836 | 48.2 |
| bidv | 0.4535 | 0.3339 | 0.2126 | 89.0 |
| mbbank | 0.4931 | 0.3076 | 0.1993 | 85.6 |
| ocb | 0.4199 | 0.3368 | 0.2433 | 42.0 |
| shb | 0.5017 | 0.2755 | 0.2228 | 86.0 |
| techcombank | 0.5466 | 0.2642 | 0.1891 | 76.6 |
| vietcombank | 0.4941 | 0.3076 | 0.1983 | 57.2 |
| viettinbank | 0.5345 | 0.2737 | 0.1918 | 82.4 |
| vpbank | 0.516 | 0.2683 | 0.2157 | 69.8 |


## RQ2 — Selective disclosure
- Share trung binh: E=0.2401 · S=0.357 · G=0.4029
- Friedman chi2=53.625, p=2.267e-12 (n_block=45)
- Wilcoxon env_vs_gov: p=2.027e-09, median_diff=-0.1596

## RQ3 — Substance gap / xu huong
- CTI~year Spearman rho=-0.2964 (p=0.04801)
- QDR~year Spearman rho=-0.0442 (p=0.7732)
- n_commit~year rho=0.6675 (p=5.496e-07)

## Validation — on dinh ranking (sensitivity)
- Bootstrap Kendall tau mean=0.7039 (p05=0.5), top-1 retention=0.521
- Leave-one-year-out tau_min=0.5916