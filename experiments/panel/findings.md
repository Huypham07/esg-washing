# ESG-washing — consolidated findings (RQ1–RQ5)

Panel: 9 banks x 5 years = 45 bank-years.

## RQ1 — Prevalence
- CTI = 0.3833 (sd 0.0832, CI95 [0.3601, 0.4077])
- NAR = 0.313 (sd 0.0876, CI95 [0.2881, 0.3367])
- QDR = 0.3037 (sd 0.0767, CI95 [0.2819, 0.3266])

## RQ2 — Selective disclosure
- pillar mean share: {'env': 0.2401, 'soc': 0.357, 'gov': 0.4029}
- Friedman chi2=53.625, p=2.267e-12

## RQ3 — Temporal trend
- CTI~year Spearman rho=-0.0865 (p=0.572)
- QDR~year Spearman rho=0.0678 (p=0.6583)

## RQ4 — Convergent validity (embedding BRI vs rubric)
- Spearman(CTI, BRI) = -0.429 (p=0.003277, n=45)
- Spearman(NAR, BRI) = 0.37 (p=0.01235, n=45)
- Interpretation: boilerplate (cross-bank reused language) tracks NAMED actions, not vague cheap-talk; rubric CTI is not redundant with embedding similarity. (SBS dropped: anisotropy.)

## RQ5 — Say-do gap + drivers
- mean say-do by pillar (CTI_p - QDR_p): {'env': 0.0551, 'soc': 0.0223, 'gov': 0.3307} -> governance is the cheap-talk pillar.
- quantified figures by type: {'other': 1289, 'green_credit': 208, 'social': 176, 'training': 146, 'trees': 27, 'energy': 26, 'emissions': 13}

## Cross-lingual transfer (lexical baseline)
- topic VI->VI: macro-F1=0.8323
- topic EN->VI: macro-F1=0.2204
- topic EN->EN: macro-F1=0.7868
- commitment VI->VI: macro-F1=0.664
- commitment EN->VI: macro-F1=0.47
- commitment EN->EN: macro-F1=0.641
- EN->VI collapse motivates translate-train / multilingual encoder.