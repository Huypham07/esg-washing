# ESG-washing — consolidated findings (RQ1–RQ5)

Panel: 9 banks x 5 years = 45 bank-years.

## RQ1 — Prevalence
- CTI = 0.372 (sd 0.0919, CI95 [0.3452, 0.3995])
- NAR = 0.3094 (sd 0.0896, CI95 [0.284, 0.3342])
- QDR = 0.3186 (sd 0.0943, CI95 [0.293, 0.3479])

## RQ2 — Selective disclosure
- pillar mean share: {'env': 0.241, 'soc': 0.3486, 'gov': 0.4104}
- Friedman chi2=50.0678, p=1.343e-11

## RQ3 — Temporal trend
- CTI~year Spearman rho=0.0309 (p=0.8405)
- QDR~year Spearman rho=-0.0732 (p=0.6327)

## RQ4 — Convergent validity (embedding BRI vs rubric)
- Spearman(CTI, BRI) = -0.3525 (p=0.01755, n=45)
- Spearman(NAR, BRI) = 0.3001 (p=0.0452, n=45)
- Interpretation: boilerplate (cross-bank reused language) tracks NAMED actions, not vague cheap-talk; rubric CTI is not redundant with embedding similarity. (SBS dropped: anisotropy.)

## RQ5 — Say-do gap + drivers
- mean say-do by pillar (CTI_p - QDR_p): {'env': 0.0255, 'soc': -0.0103, 'gov': 0.3271} -> governance is the cheap-talk pillar.
- quantified figures by type: {'other': 1259, 'green_credit': 202, 'social': 156, 'training': 141, 'trees': 27, 'energy': 23, 'emissions': 13}

## Cross-lingual transfer (lexical baseline)
- topic VI->VI: macro-F1=0.8323
- topic EN->VI: macro-F1=0.2204
- topic EN->EN: macro-F1=0.7868
- commitment VI->VI: macro-F1=0.664
- commitment EN->VI: macro-F1=0.47
- commitment EN->EN: macro-F1=0.641
- EN->VI collapse motivates translate-train / multilingual encoder.