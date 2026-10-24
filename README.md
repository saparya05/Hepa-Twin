# Hepa-Twin: A Digital Twin for Early Warning of Decompensation in Cirrhosis

**Digital Twin Challenge 2026 (Happiest Health)** | Decision support only. Synthetic data. Not for clinical use.

## 1. Team details
- Team name: `Blossom`  |  Leader: `Saparya Jagannath`  |  Email: `saparya05@gmail.com` |  Phone Number: `9311705690`
- College / Incubator: `Bharati Vidyapeeth's College of Engineering, New Delhi`  |  Members: `Solo participant`
- Folder name: `Blossom_BharatiVidyapeeth'sCollegeofEngineering`

## 2. Project title
Hepa-Twin: patient-specific digital twin that fuses EHR data and wearable time series to predict decompensation in cirrhosis 7 days ahead.

## 3. Problem statement and healthcare use case
Cirrhosis is a major chronic disease burden in India (alcohol-related disease, MASLD, hepatitis B/C), often diagnosed late, with patients far from hepatology centres. A stable patient can decompensate suddenly (hepatic encephalopathy, ascites, variceal bleed, infection, unplanned admission), and clinicians have no view of the patient between visits.
Early warning signs can show in daily-life signals: sleep-wake inversion (early encephalopathy), falling activity, rising resting heart rate, falling HRV.

**Use case:** a clinic or primary-care team triages a panel of cirrhosis patients each day; the twin flags who is likely to decompensate in the next 7 days, explains why, and lets the doctor test what-if scenarios (e.g. better lactulose adherence).

## 4. What it does

| Component | What it does | Where |
|---|---|---|
| **Patient records (static data)** | Holds each patient's slow-changing profile: age, sex, cause of cirrhosis, lab results, MELD-Na, Child-Pugh score, medicines, lactulose adherence, past decompensation and a PNPLA3 genetic marker. | `src/data_gen.py` |
| **Wearable signals (dynamic data)** | Produces daily steps, resting heart rate, HRV, sleep and night-time activity, with realistic gaps, weekly patterns and the effect of beta-blockers. | `src/data_gen.py` |
| **Digital twin** | Keeps a live state for every patient: a slow part (disease severity) and a fast part (how far today's signals are from that patient's own normal). | `src/twin.py` |
| **Early-warning model** | Combines both data streams and estimates the chance of decompensation in the next 7 days, with an uncertainty band and plain-language reasons. | `src/features.py`, `src/train.py`, `src/explain.py` |
| **Doctor dashboard** | Shows a clinic overview, each patient's risk timeline, alerts with reasons, a what-if simulator and a triage list sorted by risk. | `app/app.py` |

**Privacy by design:** every patient in the project is synthetic. No real patient data is used or stored.

## 5. Technical stack

| Layer | Tools |
|---|---|
| Language | Python 3.11+ |
| Data and features | pandas, numpy |
| Machine learning | LightGBM, scikit-learn |
| Explainability | SHAP |
| Dashboard | Streamlit, Plotly |
| Plots and reports | matplotlib |
| Model storage | joblib |
| Version control | Git / GitHub |
| License | MIT; libraries keep their own licenses |

All dependencies are listed in `requirements.txt`. Everything runs locally on a normal laptop with no GPU and no external services, and no real patient data is used.

## 6. AI/ML framework and model details

**Framework:** LightGBM (gradient-boosted decision trees) for prediction, scikit-learn for the baseline model, calibration and metrics, and SHAP for explainability.

**Task:** binary classification. The model predicts whether a decompensation event (hepatic encephalopathy, ascites worsening, variceal bleed, infection/SBP or unplanned admission) occurs within the next 7 days, using a 14-day lookback. The output is a calibrated risk (0-100%) for every patient-day.

**1. Digital twin state (`src/twin.py`).** Each patient has a persistent state updated one day at a time:
- *Slow component:* exponential moving average of MELD-Na (disease-severity trajectory).
- *Fast component:* exponentially smoothed deviation of 7 wearable signals (steps, resting HR, HRV, sleep duration, sleep efficiency, sleep midpoint, night-time activity) from the patient's own adaptive baseline.
- The baseline starts as a 7-day running mean and then adapts slowly, only on days within a normal range for that patient, so an emerging warning pattern is not absorbed into "normal". Missing wearable days leave the state unchanged.

**2. Features (`src/features.py`), 87 in total:**
- *Wearable stream (59):* rolling mean (3 and 14 days), 7-day variability, short-vs-long trend, deviation from own baseline, twin deviation, the baseline level itself, a sleep-wake inversion score, wear-time fraction and a fast twin index.
- *EHR stream (28):* demographics, etiology, alcohol abstinence, lactulose adherence, medications, rural/distance, prior decompensation, PNPLA3 genetic marker, latest labs, ascites and encephalopathy grade, Child-Pugh points, MELD-Na and its 4-week trend, slow twin index, days since last labs, prior events and days since last event.
- Days within 7 days after an event (hospital stay and recovery) are excluded from training and evaluation.

**3. Models (`src/train.py`, `src/modeling.py`):**
- *Ablation models:* wearable-only, EHR-only and fused LightGBM models, plus a fused logistic regression baseline (median imputation, standardization, C = 0.3).
- *Final model:* a bootstrap ensemble of 6 LightGBM classifiers, each trained on patients resampled with replacement. Hyperparameters: 250 trees, learning rate 0.04, 15 leaves, minimum 40 samples per leaf, row subsampling 0.8, column subsampling 0.7, L2 regularization 2.0.
- *Uncertainty:* the risk is the ensemble mean; the dashboard band is the 10th-90th percentile across ensemble members.

**4. Calibration and alert logic:**
- Isotonic regression, fitted on held-out validation patients, makes predicted risk match observed frequency.
- The alert threshold is chosen on validation patients as the lowest threshold giving at most 1 false alert per patient-month. An alert can re-fire at most once every 7 days. A false alert means no event in the following 7 days.

**5. Data split and evaluation:**
- 700 training patients (80% train, 20% validation, split by patient, never by row), a held-out test cohort of 250 patients, and a 250-patient distribution-shift cohort with higher noise, a different etiology mix, 50% silent shocks, more beta-blocker use and more missing data. Each patient has 150 days of data.
- Metrics: AUROC, AUPRC, Brier score, sensitivity at 90% specificity, share of events with an alert in the preceding 7 days, median lead time and false alerts per patient-month.

**6. Explainability and what-if:**
- SHAP (TreeExplainer) attributes each prediction to features; the top risk-increasing factors are converted to plain-language sentences (e.g. "Night-time activity +12 percentage points vs own baseline").
- The what-if simulator modifies a patient's inputs (lactulose adherence, daily steps, sleep regularity, alcohol abstinence, beta-blocker use) and re-scores them. This reflects associations learned from synthetic data, not causal effects.

## 7. Results

### Model results

| Model | AUROC (test) | AUPRC (test) | AUROC (shift) | AUPRC (shift) |
|---|---|---|---|---|
| Wearable-only | 0.724 | 0.205 | 0.657 | 0.165 |
| EHR-only | 0.746 | 0.206 | 0.717 | 0.178 |
| **Fused (EHR + wearable)** | **0.776** | **0.263** | 0.717 | 0.191 |
| Logistic regression (fused) | 0.776 | 0.267 | 0.731 | 0.215 |

### Fused ensemble results

| Fused ensemble (calibrated) | Test | Shift |
|---|---|---|
| AUROC / AUPRC | 0.783 / 0.258 | 0.731 / 0.198 |
| Brier score | 0.059 | 0.071 |
| Sensitivity at 90% specificity | 0.46 | 0.25 |
| Events with an alert in the prior 7 days | 64% | 79% |
| Median lead time | 4 days | 4 days |
| False alerts / patient-month | 0.65 | 1.83 |

## 8. Dashboard
`streamlit run app/app.py`: patient summary, risk timeline with uncertainty band and event markers, wearables vs own baseline, labs and medication, alert with plain-language reasons, what-if sliders, clinic triage list, permanent disclaimer banner. 
<img width="1049" height="538" alt="image" src="https://github.com/user-attachments/assets/50e337a2-37bb-42c9-ae1c-90285ad45c22" />


## 9. Ethics, privacy, DPDP / HIPAA note
Only synthetic data. No personal data is collected or stored. The tool is decision support and must never replace clinical judgment. Any real deployment would need ethical approval, consent, local-first processing, validation on real cohorts, bias and calibration audits, and DPDP Act compliance.

## 10. How to run
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.data_gen                                 # synthetic cohorts -> data/        
python -m src.train                                    # features, ablation, model -> models/, results/   
streamlit run app/app.py                               # dashboard
python docs/make_architecture.py                       # architecture diagram -> docs/
```
The test/shift cohorts and the trained model are pre-generated so the dashboard works immediately; the (large) training cohort is regenerated by `python -m src.data_gen` with fixed seeds.

## 11. Deliverables and links
- Architecture diagram: https://github.com/saparya05/Hepa-Twin/blob/main/docs/architecture.pdf
- Presentation (PDF/PPT): https://github.com/saparya05/Hepa-Twin/blob/main/docs/Hepa-Twin_Presentation.pdf
- Video (unlisted YouTube): `<ADD LINK>`

## 12. License
This project is open-source and available under the MIT License.
