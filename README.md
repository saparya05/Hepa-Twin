# Hepa-Twin: A Digital Twin for Early Warning of Decompensation in Cirrhosis

**Digital Twin Challenge 2026 (Happiest Health)** | Decision support only. Synthetic data. Not for clinical use.

## 1. Team details
- Team name: `Blossom`  |  Leader: `Saparya Jagannath`  |  Email: `saparya05@gmail.com` |  Phone Number: `9311705690`
- College / Incubator: `Bharati Vidyapeeth's College of Engineering, New Delhi`  |  Members: Solo participant
- Folder name for submission: `Blossom_BharatiVidyapeeth'sCollegeofEngineering`

## 2. Project title
Hepa-Twin: patient-specific digital twin that fuses EHR data and wearable time series to predict decompensation in cirrhosis 7 days ahead.

## 3. Problem statement and healthcare use case
Cirrhosis is a major chronic disease burden in India (alcohol-related disease, MASLD, hepatitis B/C), often diagnosed late, with patients far from hepatology centres. A stable patient can decompensate suddenly (hepatic encephalopathy, ascites, variceal bleed, infection, unplanned admission), and clinicians have no view of the patient between visits.
Early warning signs can show in daily-life signals: sleep-wake inversion (early encephalopathy), falling activity, rising resting heart rate, falling HRV.

**Use case:** a clinic or primary-care team triages a panel of cirrhosis patients each day; the twin flags who is likely to decompensate in the next 7 days, explains why, and lets the doctor test what-if scenarios (e.g. better lactulose adherence).

## 4. What it does (requirements mapping)
| Hackathon requirement | Where |
|---|---|
| Static stream: simulated EHR (demographics, diagnoses, labs, genetic marker) | `src/data_gen.py` (labs, MELD-Na, Child-Pugh, meds, etiology, history, PNPLA3 rs738409 risk-allele count) |
| Dynamic stream: wearable time series | `src/data_gen.py` (steps, resting HR, HRV, sleep, night activity, sleep midpoint; missing data, weekly pattern, beta-blocker effect) |
| Fusion + predict adverse event in advance | `src/features.py`, `src/train.py` (7-day decompensation risk) |
| Persistent twin | `src/twin.py` (slow severity + fast deviation state per patient) |
| Conceptual doctor dashboard | `app/app.py` (Streamlit) |
| Anonymised / synthetic data only | everything is generated; no real patient data |

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
| License | MIT (code); libraries keep their own licenses |

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

**7. Limitations of the modelling approach:** all data are synthetic, so the results demonstrate the method, not clinical validity. The relationship between wearables and events is built into the generator (with 35% silent shocks to limit circularity), and the genetic marker and its effect size are assumptions.

## 7. Data and synthetic methodology
No real patient data is used. A hidden "hepatic reserve" drifts with etiology, abstinence and adherence; random shocks (infection, encephalopathy, bleed, dehydration, alcohol relapse) ramp up over 3-10 days. **Events are drawn from a hazard of the hidden state, MELD-Na and adherence, not from wearable values.** Wearables reflect a shock only if it has a visible prodrome; **35% of shocks are silent** (50% in the shift cohort), so some events cannot be predicted from wearables.
Optional realism checks against open data (verify licences yourself): UCI Indian Liver Patient Dataset (lab ranges), UCI Cirrhosis Patient Survival Prediction (Mayo PBC trial; not representative of alcohol/MASLD cirrhosis). MIMIC-IV was deliberately not used.
**Genetic marker:** each synthetic patient carries 0-2 PNPLA3 rs738409 risk alleles (assumed frequency 0.35, small assumed effect on starting liver reserve). Both numbers are assumptions, not findings; verify against literature before citing.
**Limitation (circularity):** because signals and labels are both generated by us, results demonstrate the pipeline and method, **not clinical validity**. Clinical constants are assumptions. MELD-Na uses the 2016 UNOS formula (MELD 3.0 exists).

## 8. Results
Numbers below are from one run with the default seeds (`python -m src.train`; regenerate and paste your own from `results/ablation.md`).

| Model | AUROC (test) | AUPRC (test) | AUROC (shift) | AUPRC (shift) |
|---|---|---|---|---|
| Wearable-only | 0.724 | 0.205 | 0.657 | 0.165 |
| EHR-only | 0.746 | 0.206 | 0.717 | 0.178 |
| **Fused (EHR + wearable)** | **0.776** | **0.263** | 0.717 | 0.191 |
| Logistic regression (fused) | 0.776 | 0.267 | 0.731 | 0.215 |

| Fused ensemble (calibrated) | Test | Shift |
|---|---|---|
| AUROC / AUPRC | 0.783 / 0.258 | 0.731 / 0.198 |
| Brier score | 0.059 | 0.071 |
| Sensitivity @ 90% specificity | 0.46 | 0.25 |
| Events with an alert in the prior 7 days | 64% | 79% |
| Median lead time | 4 days | 4 days |
| False alerts / patient-month | 0.65 | 1.83 |

## 9. Dashboard
`streamlit run app/app.py`: patient summary, risk timeline with uncertainty band and event markers, wearables vs own baseline, labs and medication, alert with plain-language reasons, what-if sliders, clinic triage list, permanent disclaimer banner. 
<img width="1049" height="538" alt="image" src="https://github.com/user-attachments/assets/50e337a2-37bb-42c9-ae1c-90285ad45c22" />


## 10. Ethics, privacy, DPDP / HIPAA note
Only synthetic data. No personal data is collected or stored. The tool is decision support and must never replace clinical judgment. Any real deployment would need ethical approval, consent, local-first processing, validation on real cohorts, bias and calibration audits, and DPDP Act compliance.

## 11. How to run
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.data_gen        # synthetic cohorts -> data/        
python -m src.train           # features, ablation, model -> models/, results/   
streamlit run app/app.py      # dashboard
python docs/make_architecture.py   # architecture diagram -> docs/
```
The test/shift cohorts and the trained model are pre-generated so the dashboard works immediately; the (large) training cohort is regenerated by `python -m src.data_gen` with fixed seeds.

## 12. Deliverables and links
- Architecture diagram: `docs/architecture.pdf`
- Presentation (PDF/PPT): `docs/<add>`
- Video (20+ min, unlisted YouTube): `<ADD LINK>`
- License: MIT (code). Libraries keep their own licences. No external datasets are bundled.
