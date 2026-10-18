| Model | AUROC (test) | AUPRC (test) | AUROC (shift) | AUPRC (shift) |
|---|---|---|---|---|
| Wearable-only | 0.724 | 0.205 | 0.657 | 0.165 |
| EHR-only | 0.746 | 0.206 | 0.717 | 0.178 |
| Fused (EHR + wearable) | 0.776 | 0.263 | 0.717 | 0.191 |
| Logistic regression (fused) | 0.776 | 0.267 | 0.731 | 0.215 |

| Fused ensemble (calibrated) | Test | Shift |
|---|---|---|
| AUROC | 0.783 | 0.731 |
| AUPRC | 0.258 | 0.198 |
| Brier score | 0.059 | 0.071 |
| Sensitivity @ 90% specificity | 0.46 | 0.25 |
| Events detected (alert in prior 7 d) | 64% | 79% |
| Median lead time (days) | 4.0 | 4.0 |
| False alerts / patient-month | 0.65 | 1.83 |

Alert threshold: 11.1% risk (chosen on validation patients for <= 1.0 false alerts per patient-month).
