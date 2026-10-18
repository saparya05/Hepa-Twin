import argparse, os
import numpy as np
import pandas as pd

ETIOLOGIES = ["alcohol", "masld", "hbv", "hcv", "mixed"]
# signature per shock: (steps %, HR bpm, HRV %, sleep efficiency, sleep-midpoint shift h, night-activity fraction)
SIGARR = np.array([
    [-0.35, 10, -0.30, -0.08, 0.5, 0.03],   # 0 infection
    [-0.20,  1, -0.10, -0.12, 3.0, 0.20],   # 1 encephalopathy (sleep-wake inversion)
    [-0.40, 18, -0.35, -0.05, 0.3, 0.02],   # 2 bleed
    [-0.25,  9, -0.20, -0.04, 0.2, 0.02],   # 3 dehydration
    [-0.15,  6, -0.20, -0.10, 1.5, 0.10],   # 4 alcohol relapse
    [0, 0, 0, 0, 0, 0],                      # index -1 -> no shock
], dtype=float)
EVENT_NAME = {0: "Infection / SBP", 1: "Hepatic encephalopathy", 2: "Variceal bleed",
              3: "Unplanned admission", 4: "Ascites worsening"}
DEFAULT_CFG = dict(etiology_p=[0.40, 0.25, 0.15, 0.12, 0.08], noise=1.0, silent_frac=0.35,
                   beta_blocker_p=0.60, shock_rate=0.006, missing_p=0.06)
SHIFT_CFG = dict(etiology_p=[0.25, 0.40, 0.15, 0.10, 0.10], noise=1.5, silent_frac=0.50,
                 beta_blocker_p=0.80, shock_rate=0.005, missing_p=0.12)
T_DEFAULT = 150
START = pd.Timestamp("2026-03-01")


def meld_na(bili, inr, cr, na):
    """MELD-Na (UNOS 2016 version). MELD 3.0 exists; verify which one you cite."""
    bili, inr, cr, na = max(bili, 1.0), max(inr, 1.0), min(max(cr, 1.0), 4.0), min(max(na, 125.0), 137.0)
    meld = round(10 * (0.957 * np.log(cr) + 0.378 * np.log(bili) + 1.120 * np.log(inr) + 0.643))
    m = meld + 1.32 * (137 - na) - 0.033 * meld * (137 - na)
    return float(min(max(m, 6), 40))


def child_pugh(bili, alb, inr, asc, he):
    pts = (1 if bili < 2 else 2 if bili <= 3 else 3) + (1 if alb > 3.5 else 2 if alb >= 2.8 else 3) \
        + (1 if inr < 1.7 else 2 if inr <= 2.3 else 3) + (asc + 1) + (he + 1)
    return int(pts)


def labs_from_reserve(r, rng, noise):
    d = float(np.clip(1 - r, 0, 1))
    ln = lambda s: rng.lognormal(0, s * noise)
    bili = 0.7 + 7.0 * d ** 2 * ln(0.25)
    inr = 1.0 + 1.6 * d ** 1.5 * ln(0.08)
    cr = 0.8 + 1.8 * d ** 2.5 * ln(0.20)
    na = 139 - 15 * d ** 2 + rng.normal(0, 1.2 * noise)
    alb = 4.3 - 1.8 * d + rng.normal(0, 0.2 * noise)
    plt = max(25, 220 - 160 * d + rng.normal(0, 20 * noise))
    dd = d + rng.normal(0, 0.08)
    asc = int(dd > 0.38) + int(dd > 0.62)
    dh = d + rng.normal(0, 0.08)
    he = int(dh > 0.50) + int(dh > 0.75)
    return bili, inr, cr, na, alb, plt, asc, he


def ar1(rng, T, phi, sd):
    e = rng.normal(0, sd, T)
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = phi * x[t - 1] + e[t]
    return x


def make_static(rng, cfg):
    et = str(rng.choice(ETIOLOGIES, p=cfg["etiology_p"]))
    alc = et in ("alcohol", "mixed")
    prior = int(rng.random() < 0.30)
    rural = int(rng.random() < 0.55)
    # Genetic marker (static EHR field). PNPLA3 rs738409 risk-allele count (0/1/2), linked in the literature to
    # fatty-liver / alcohol-related liver disease. ASSUMED frequency 0.35 and a small ASSUMED effect: verify before citing.
    pnpla3 = int(rng.binomial(2, 0.35))
    p = dict(
        etiology=et, age=int(np.clip(rng.normal(52, 11), 28, 80)),
        sex="M" if rng.random() < (0.80 if alc else 0.60) else "F",
        bmi=float(np.clip(rng.normal(28.5 if et == "masld" else 24.5, 3.5), 16, 42)),
        alcohol_abstinent=int(rng.random() < 0.5) if alc else 1,   # 1 = abstinent OR not applicable
        lactulose_adherence=float(np.clip(rng.beta(4, 2), 0.1, 1.0)),
        beta_blocker=int(rng.random() < cfg["beta_blocker_p"]),
        diuretic=int(rng.random() < 0.55), rifaximin=int(rng.random() < 0.20),
        rural=rural, distance_km=float(np.clip(rng.lognormal(3.6 if rural else 2.6, 0.6), 3, 400)),
        prior_decomp=prior, pnpla3_risk=pnpla3,
    )
    p["r0"] = float(np.clip(rng.beta(4, 2.5) * 0.8 + 0.15 - 0.10 * prior - 0.03 * pnpla3, 0.20, 0.92))   # small assumed genetic effect on starting reserve
    # hidden device baselines (not saved)
    p["_steps0"] = float(np.clip(2500 + 4500 * p["r0"] - 25 * (p["age"] - 55) + rng.normal(0, 500), 800, 9000))
    p["_hr0"] = 74 + 6 * (1 - p["r0"]) - 10 * p["beta_blocker"] + rng.normal(0, 5)
    p["_hrv0"] = float(np.exp(rng.normal(np.log(28 + 10 * p["r0"]), 0.2)) * (1.1 if p["beta_blocker"] else 1.0))
    p["_dur0"] = 6.8 + rng.normal(0, 0.5)
    p["_eff0"] = float(np.clip(0.86 - 0.05 * (1 - p["r0"]) + rng.normal(0, 0.03), 0.6, 0.95))
    p["_mid0"] = 2.5 + rng.normal(0, 0.6)
    p["_night0"] = float(np.clip(0.06 + 0.06 * (1 - p["r0"]) + rng.normal(0, 0.015), 0.02, 0.2))
    return p


def simulate_patient(pid, p, T, rng, cfg):
    noise = cfg["noise"]
    r = p["r0"]
    adh, abst, prior = p["lactulose_adherence"], p["alcohol_abstinent"], p["prior_decomp"]
    u_shock, u_event = rng.random(T), rng.random(T)
    e = np.zeros(T); stype = np.full(T, -1); prod = np.zeros(T); reserve = np.zeros(T)
    event = np.zeros(T, int); ev_names = {}
    labs = np.zeros((T, 8)); meld = np.zeros(T); cpp = np.zeros(T); dsv = np.zeros(T)
    shock, refractory, next_visit, last_visit, cur, cur_meld, cur_cp = None, 0, 0, 0, None, 0.0, 0
    for t in range(T):
        refractory -= 1
        r += -(0.0004 + 0.0008 * (1 - abst) + 0.0006 * (1 - adh)) + rng.normal(0, 0.0015)
        r = float(min(max(r, 0.05), 0.98)); reserve[t] = r
        if shock is None and refractory <= 0 and t >= 3:
            rate = cfg["shock_rate"] * (1 + 1.5 * (1 - adh) + 1.2 * (1 - abst) + 0.8 * prior + 1.5 * (1 - r))
            if u_shock[t] < rate:
                w = np.array([0.25, 0.22 * (1 + 2 * (1 - adh)), 0.10 * (1 + prior), 0.15 * (1 + p["diuretic"]),
                              0.20 * (1 - abst)])
                shock = dict(type=int(rng.choice(5, p=w / w.sum())), onset=t, m=rng.uniform(0.6, 1.2),
                             L=int(rng.integers(3, 11)), hold=int(rng.integers(2, 5)),
                             prod=int(rng.random() > cfg["silent_frac"]))
        if shock is not None:
            k, L, h = t - shock["onset"], shock["L"], shock["hold"]
            if k < L: val = shock["m"] * (k + 1) / L
            elif k < L + h: val = shock["m"]
            elif k < L + h + 4: val = shock["m"] * (1 - (k - L - h + 1) / 4)
            else: shock, val = None, 0.0
            if shock is not None:
                e[t], stype[t], prod[t] = val, shock["type"], shock["prod"]
        if t >= next_visit:  # clinic visit: labs refresh
            cur = labs_from_reserve(r - 0.12 * e[t], rng, noise)
            cur_meld = meld_na(cur[0], cur[1], cur[2], cur[3])
            cur_cp = child_pugh(cur[0], cur[4], cur[1], cur[6], cur[7])
            last_visit, next_visit = t, t + int(rng.integers(14, 29))
        labs[t], meld[t], cpp[t], dsv[t] = cur, cur_meld, cur_cp, t - last_visit
        # event HAZARD: depends on hidden state + EHR, NOT directly on wearable values
        logit = -7.6 + 3.0 * (1 - r) + 0.06 * (cur_meld - 10) - 0.8 * adh + 0.7 * prior + 4.5 * e[t]
        if refractory <= 0 and t >= 7 and u_event[t] < 1 / (1 + np.exp(-logit)):
            event[t] = 1
            ev_names[t] = EVENT_NAME[shock["type"]] if shock else str(rng.choice(["Ascites worsening", "Unplanned admission"]))
            r = max(r - 0.03, 0.05); shock = None; refractory = 10; next_visit = min(next_visit, t + 5)

    w = e * prod                                     # only 'visible' shocks perturb the wearables
    sg = SIGARR[stype]
    weekend = np.where(np.isin(np.arange(T) % 7, [5, 6]), 0.90, 1.0)
    blunt = 0.5 if p["beta_blocker"] else 1.0        # beta-blocker blunts HR response
    steps = p["_steps0"] * (1 + 0.5 * (reserve - p["r0"])) * weekend * (1 + sg[:, 0] * w) * np.exp(ar1(rng, T, 0.6, 0.10 * noise))
    hr = p["_hr0"] + sg[:, 1] * w * blunt + ar1(rng, T, 0.5, 2.0 * noise)
    hrv = p["_hrv0"] * (1 + sg[:, 2] * w) * np.exp(ar1(rng, T, 0.5, 0.10 * noise))
    eff = np.clip(p["_eff0"] + sg[:, 3] * w + rng.normal(0, 0.03 * noise, T), 0.4, 0.98)
    dur = np.clip(p["_dur0"] + 8 * sg[:, 3] * w + rng.normal(0, 0.5 * noise, T), 3, 11)
    mid = p["_mid0"] + sg[:, 4] * w + ar1(rng, T, 0.4, 0.3 * noise)
    night = np.clip(p["_night0"] + sg[:, 5] * w + rng.normal(0, 0.015 * noise, T), 0.01, 0.6)
    miss = rng.random(T) < cfg["missing_p"]                      # device not worn / battery
    for s in np.where(rng.random(T) < 0.01)[0]: miss[s:s + int(rng.integers(2, 6))] = True
    for ed in np.where(event == 1)[0]: miss[ed + 1: ed + 5] = True   # hospital stay: no wearable data
    W = np.column_stack([steps, hr, hrv, dur, eff, mid, night]); W[miss] = np.nan
    df = pd.DataFrame(W, columns=["steps", "rest_hr", "hrv", "sleep_dur", "sleep_eff", "mid", "night"])
    df.insert(0, "day", np.arange(T)); df.insert(0, "patient_id", pid)
    df["date"] = (START + pd.to_timedelta(df["day"], unit="D")).dt.strftime("%Y-%m-%d")
    for i, c in enumerate(["bili", "inr", "creat", "sodium", "albumin", "platelets", "ascites", "he_grade"]):
        df[c] = labs[:, i]
    df["meld_na"], df["child_pugh_pts"], df["days_since_visit"] = meld, cpp, dsv
    df["event"] = event
    df["latent_reserve"], df["latent_shock"], df["latent_prodrome_visible"] = reserve, e, prod  # NEVER used as features
    return df, [(pid, int(d), ev_names[d]) for d in ev_names]


def generate_cohort(n, seed, cfg, T=T_DEFAULT, prefix="P"):
    rng = np.random.default_rng(seed)
    pats, dailies, evs = [], [], []
    for i in range(n):
        pid = f"{prefix}{i:04d}"
        p = make_static(rng, cfg)
        d, ev = simulate_patient(pid, p, T, rng, cfg)
        pats.append({"patient_id": pid, **{k: v for k, v in p.items() if not k.startswith("_")}})
        dailies.append(d); evs += ev
    return (pd.DataFrame(pats), pd.concat(dailies, ignore_index=True),
            pd.DataFrame(evs, columns=["patient_id", "day", "event_type"]))


def main(out="data"):
    os.makedirs(out, exist_ok=True)
    for name, n, seed, cfg, prefix in [("train", 700, 11, DEFAULT_CFG, "T"), ("test", 250, 22, DEFAULT_CFG, "H"),
                                       ("shift", 250, 33, SHIFT_CFG, "S")]:
        pats, daily, ev = generate_cohort(n, seed, cfg, prefix=prefix)
        pats.to_csv(f"{out}/{name}_patients.csv", index=False)
        daily.to_csv(f"{out}/{name}_daily.csv", index=False)
        ev.to_csv(f"{out}/{name}_events.csv", index=False)
        print(f"{name}: {n} patients, {len(daily)} patient-days, {len(ev)} events ({len(ev)/n:.2f}/patient)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data"); main(ap.parse_args().out)
