import numpy as np
import shap

LABEL = {"steps": "Daily steps", "rest_hr": "Resting heart rate", "hrv": "Heart-rate variability",
         "sleep_dur": "Sleep duration", "sleep_eff": "Sleep efficiency",
         "mid": "Sleep midpoint", "night": "Night-time activity"}
_cache = {}


def _explainer(bundle):
    key = id(bundle["models"][0])
    if key not in _cache:
        _cache[key] = shap.TreeExplainer(bundle["models"][0])
    return _cache[key]


def shap_row(bundle, X_row):
    sv = _explainer(bundle).shap_values(X_row[bundle["feature_cols"]])
    if isinstance(sv, list): sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3: sv = sv[:, :, 1]
    return sv[0]


def _sentence(feat, v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return None
    for m, name in LABEL.items():
        for suf in ("twin", "dev3", "dev7", "slope", "m3", "m14", "sd7", "base"):
            if feat == f"{m}_{suf}":
                if suf in ("twin", "dev3", "dev7"):
                    if m in ("steps", "rest_hr", "hrv", "sleep_dur", "sleep_eff"):
                        return f"{name} {abs(v)*100:.0f}% {'below' if v < 0 else 'above'} this patient's own baseline"
                    if m == "mid": return f"Sleep midpoint shifted {v:+.1f} h vs own baseline"
                    return f"Night-time activity {v*100:+.0f} percentage points vs own baseline"
                if suf == "sd7": return f"{name} is unusually variable this week"
                if suf == "slope": return f"{name} trending away from its 2-week level"
                return f"{name} recent level ({v:.2f}) is informative"
    table = {"inversion": f"Sleep-wake inversion score elevated ({v:.2f})",
             "twin_fast": f"Fast twin component (wearable deviation index) elevated ({v:.2f})",
             "twin_slow": f"Slow twin component (smoothed MELD-Na) is {v:.0f}",
             "meld_na": f"MELD-Na is {v:.0f}", "meld_trend": f"MELD-Na changed {v:+.1f} over ~4 weeks",
             "lactulose_adherence": f"Lactulose adherence is {v*100:.0f}%",
             "alcohol_abstinent": "Not abstinent from alcohol" if v < 0.5 else "Abstinent from alcohol",
             "prior_decomp": "Previous decompensation episode" if v > 0.5 else "No prior decompensation",
             "n_events_prior": f"{int(v)} prior event(s) in the observation period",
             "days_since_event": f"{int(v)} days since last event" if v < 900 else "No recent event",
             "days_since_visit": f"{int(v)} days since last clinic visit/labs",
             "child_pugh_pts": f"Child-Pugh score {int(v)} points", "ascites": f"Ascites grade {int(v)}",
             "he_grade": f"Encephalopathy grade {int(v)}", "bili": f"Bilirubin {v:.1f} mg/dL",
             "inr": f"INR {v:.2f}", "creat": f"Creatinine {v:.2f} mg/dL", "sodium": f"Sodium {v:.0f} mmol/L",
             "albumin": f"Albumin {v:.1f} g/dL", "platelets": f"Platelets {v:.0f} x10^3/uL",
             "beta_blocker": "On a beta-blocker (blunts HR signals)" if v > 0.5 else "Not on a beta-blocker",
             "pnpla3_risk": f"Carries {int(v)} PNPLA3 risk allele(s) (genetic marker)",
             "wear_frac14": f"Wearable worn {v*100:.0f}% of last 14 days"}
    return table.get(feat, f"{feat.replace('_', ' ')} = {v:.2f}")


def top_reasons(bundle, X_row, k=4):
    """Top risk-INCREASING factors (positive SHAP) as (sentence, shap_value). Deduplicated by wearable metric."""
    sv = shap_row(bundle, X_row)
    cols = bundle["feature_cols"]
    order = np.argsort(-sv)
    out, seen = [], set()
    for i in order:
        if sv[i] <= 0 or len(out) >= k: break
        f = cols[i]
        group = f.split("_")[0] if f.rsplit("_", 1)[-1] in ("twin", "dev3", "dev7", "slope", "m3", "m14", "sd7", "base") else f
        group = f.rsplit("_", 1)[0] if group != f else f
        if group in seen: continue
        s = _sentence(f, float(X_row.iloc[0][f]))
        if s: out.append((s, float(sv[i]))); seen.add(group)
    return out
