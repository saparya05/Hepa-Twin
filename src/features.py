import numpy as np
import pandas as pd
from .twin import run_twin, METRICS

WEAR = list(METRICS)
REL = {m for m, (k, _) in METRICS.items() if k == "rel"}
SUFFIX = ["m3", "m14", "sd7", "slope", "dev3", "dev7", "twin", "base"]
WEARABLE_COLS = [f"{m}_{s}" for m in WEAR for s in SUFFIX] + ["inversion", "wear_frac14", "twin_fast"]
STATIC_COLS = ["age", "bmi", "sex_male", "etiology_code", "alcohol_abstinent", "lactulose_adherence",
               "beta_blocker", "diuretic", "rifaximin", "rural", "distance_km", "prior_decomp", "pnpla3_risk"]
LAB_COLS = ["bili", "inr", "creat", "sodium", "albumin", "platelets", "ascites", "he_grade",
            "child_pugh_pts", "meld_na", "days_since_visit", "meld_trend", "twin_slow",
            "n_events_prior", "days_since_event"]
EHR_COLS = STATIC_COLS + LAB_COLS
ALL_COLS = EHR_COLS + WEARABLE_COLS
ET_CODE = {"alcohol": 0, "masld": 1, "hbv": 2, "hcv": 3, "mixed": 4}
HORIZON, LOOKBACK, RECOVERY = 7, 14, 7


def _one_patient(g, prow):
    g = g.sort_values("day").reset_index(drop=True)
    T = len(g)
    tw = run_twin(g)
    out = {"patient_id": g["patient_id"], "day": g["day"]}
    for m in WEAR:
        s = g[m]
        m3, m7 = s.rolling(3, min_periods=1).mean(), s.rolling(7, min_periods=2).mean()
        m14 = s.rolling(14, min_periods=4).mean()
        base = tw[f"base_{m}"]
        out[f"{m}_m3"], out[f"{m}_m14"] = m3, m14
        out[f"{m}_sd7"] = s.rolling(7, min_periods=3).std()
        out[f"{m}_slope"] = m3 - m14
        out[f"{m}_dev3"] = m3 / base - 1 if m in REL else m3 - base
        out[f"{m}_dev7"] = m7 / base - 1 if m in REL else m7 - base
        out[f"{m}_twin"], out[f"{m}_base"] = tw[f"twin_dev_{m}"], base
    # sleep-wake inversion score: night-time activity above own baseline + sleep-midpoint delay (hours/3)
    out["inversion"] = np.clip(out["night_dev3"] * 5 + np.clip(out["mid_dev3"], 0, None) / 3, 0, None)
    out["wear_frac14"] = g["steps"].notna().astype(float).rolling(14, min_periods=1).mean()
    out["twin_fast"], out["twin_slow"] = tw["twin_fast"], tw["twin_slow"]
    f = pd.DataFrame(out)
    for c in ["bili", "inr", "creat", "sodium", "albumin", "platelets", "ascites", "he_grade",
              "child_pugh_pts", "meld_na", "days_since_visit", "event"]:
        f[c] = g[c].values
    f["meld_trend"] = g["meld_na"] - g["meld_na"].shift(28)
    ev = g["event"].values.astype(int)
    f["n_events_prior"] = np.cumsum(ev) - ev
    last, dse = -999, np.zeros(T)
    for t in range(T):
        if ev[t]: last = t
        dse[t] = t - last if last >= 0 else 999
    f["days_since_event"] = dse
    cs = np.concatenate([[0], np.cumsum(ev)])
    f["label"] = [(cs[min(t + HORIZON + 1, T)] - cs[t + 1]) > 0 for t in range(T)]
    f["label"] = f["label"].astype(int)
    f["valid"] = (f["day"] >= LOOKBACK) & (f["days_since_event"] >= RECOVERY) & (f["day"] <= T - 1 - HORIZON)
    for c in STATIC_COLS:
        if c == "sex_male": f[c] = float(prow["sex"] == "M")
        elif c == "etiology_code": f[c] = ET_CODE[prow["etiology"]]
        else: f[c] = float(prow[c])
    f["etiology"], f["date"] = prow["etiology"], g["date"].values
    return f


def build_features(patients: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    prows = patients.set_index("patient_id")
    parts = [_one_patient(g, prows.loc[pid]) for pid, g in daily.groupby("patient_id", sort=True)]
    df = pd.concat(parts, ignore_index=True)
    return df[df["valid"]].drop(columns="valid").reset_index(drop=True)
