import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.explain import top_reasons
from src.modeling import predict_risk
from src.whatif import apply_changes

st.set_page_config(page_title="Hepa-Twin", page_icon="🩺", layout="wide")
DATA, MODEL = ROOT / "data", ROOT / "models" / "hepatwin.joblib"
ACTIONS = {"Hepatic encephalopathy": "Phone the caregiver today; check lactulose use and bowel movements; consider early review.",
           "default": "Contact the patient/caregiver within 24 h, review medication adherence, hydration and infection signs; consider early clinic or ED review."}


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL)


@st.cache_data
def load_cohort(name):
    scored = pd.read_pickle(DATA / f"scored_{name}.pkl")
    daily = pd.read_csv(DATA / f"{name}_daily.csv")
    events = pd.read_csv(DATA / f"{name}_events.csv")
    pats = pd.read_csv(DATA / f"{name}_patients.csv").set_index("patient_id")
    return scored, daily, events, pats


def cp_class(pts):
    return "A" if pts <= 6 else "B" if pts <= 9 else "C"


PAGES = ["Home", "Patient twin", "Clinic triage", "About & limitations"]
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1250px;}
.kpi {background:#ffffff; border:1px solid #e5e7eb; border-radius:14px; padding:16px 18px; box-shadow:0 1px 3px rgba(0,0,0,.06);}
.kpi .kl {color:#6b7280; font-size:.82rem; font-weight:600; text-transform:uppercase; letter-spacing:.04em;}
.kpi .kv {font-size:2rem; font-weight:700; line-height:1.25;}
.kpi .ks {color:#6b7280; font-size:.82rem;}
.pcard {background:#ffffff; color:#1f2937; border:1px solid #e5e7eb; border-left:6px solid var(--c); border-radius:12px;
        padding:12px 14px; margin-bottom:6px; box-shadow:0 1px 3px rgba(0,0,0,.06);}
.pcard .top {display:flex; justify-content:space-between; align-items:center;}
.pcard .pid {font-weight:700; font-size:1.05rem;}
.badge {color:#fff; background:var(--c); border-radius:999px; padding:2px 10px; font-size:.75rem; font-weight:700;}
.pcard .risk {font-size:1.7rem; font-weight:800; color:var(--c);}
.pcard .meta {color:#6b7280; font-size:.82rem;}
.pcard .why {color:#374151; font-size:.85rem; margin-top:4px;}
.feed {background:#ffffff; color:#1f2937; border:1px solid #e5e7eb; border-radius:10px; padding:8px 12px; margin-bottom:6px; font-size:.88rem;}
</style>
""", unsafe_allow_html=True)


def kpi(label, value, sub="", color="#2563eb"):
    return (f'<div class="kpi"><div class="kl">{label}</div><div class="kv" style="color:{color}">{value}</div>'
            f'<div class="ks">{sub}</div></div>')


def tone(risk, thr):
    return ("#dc2626", "ALERT") if risk >= thr else (("#f59e0b", "WATCH") if risk >= 0.6 * thr else ("#16a34a", "STABLE"))


def open_patient(p):
    st.session_state["pid_sel"] = p
    st.session_state["page"] = "Patient twin"


@st.cache_data
def risk_pivot(name):
    sc = load_cohort(name)[0]
    return sc.pivot(index="day", columns="patient_id", values="risk").ffill()


if not MODEL.exists() or not (DATA / "scored_test.pkl").exists():
    st.error("Model not found. Run `python -m src.train` first (about 2-3 minutes), then reload this page.")
    st.stop()

bundle = load_bundle()
THR = bundle["threshold"] * 100

st.title(" Hepa-Twin")
st.header("Cirrhosis decompensation early-warning twin")

with st.sidebar:
    st.header("Hepa-Twin")
    page = st.radio("Go to", PAGES, key="page")
    st.divider()
    st.caption("Clinic settings")
    cohort = st.selectbox("Cohort", ["Held-out test cohort", "Distribution-shift cohort (harder)"])
    name = "test" if cohort.startswith("Held") else "shift"
    scored, daily, events, pats = load_cohort(name)
    day = st.slider("Today (simulated day)", int(scored.day.min()), int(scored.day.max()), 70)
    reveal = st.checkbox("Reveal synthetic ground truth (future)", value=False,
                         help="A doctor would not see the future. Turn on to compare predictions with what happened.")
    snap = (scored[scored.day <= day].sort_values("day").groupby("patient_id").tail(1)
            .sort_values("risk", ascending=False))
    pid = st.selectbox("Patient (highest risk first)", snap.patient_id.tolist(), key="pid_sel",
                       format_func=lambda p: f"{p}  -  risk {snap.set_index('patient_id').loc[p, 'risk']:.0f}%")
    st.caption(f"Alert threshold: {THR:.0f}% (tuned for about {bundle['target_fa']:.0f} false alert per patient-month)")


# ------------------------------------------------------------------------------------------------ home
if page == "Home":
    dsel = scored.loc[scored.day == day, "date"]
    st.subheader("Clinic overview")
    st.caption(f"Simulated day {day}" + (f"  |  {dsel.iloc[0]}" if len(dsel) else "") + f"  |  {cohort}")
    n_alert = int((snap.risk >= THR).sum()); top = snap.iloc[0]
    k = st.columns(4)
    k[0].markdown(kpi("Patients monitored", len(snap), "cirrhosis panel"), unsafe_allow_html=True)
    k[1].markdown(kpi("Alerts today", n_alert, f"threshold {THR:.0f}% risk", "#dc2626" if n_alert else "#16a34a"), unsafe_allow_html=True)
    k[2].markdown(kpi("Highest risk", f"{top.risk:.0f}%", f"{top.patient_id}  |  MELD-Na {top.meld_na:.0f}", "#dc2626"), unsafe_allow_html=True)
    k[3].markdown(kpi("Average risk", f"{snap.risk.mean():.0f}%", "across the panel", "#2563eb"), unsafe_allow_html=True)

    st.markdown("### Needs attention now")
    cards = snap.head(6)
    cols = st.columns(3)
    for i, (idx, r) in enumerate(cards.iterrows()):
        color, label = tone(r.risk, THR)
        why = top_reasons(bundle, scored.loc[[idx]], k=1)
        why_txt = why[0][0] if why else "No dominant risk factor"
        with cols[i % 3]:
            st.markdown(f'''<div class="pcard" style="--c:{color}"><div class="top"><span class="pid">{r.patient_id}</span>
                <span class="badge">{label}</span></div><div class="risk">{r.risk:.0f}%</div>
                <div class="meta">{r.etiology.upper()} | age {int(r.age)} | MELD-Na {r.meld_na:.0f} | Child-Pugh {cp_class(r.child_pugh_pts)}</div>
                <div class="why">{why_txt}</div></div>''', unsafe_allow_html=True)
            st.button("Open patient", key=f"open_{r.patient_id}", on_click=open_patient, args=(r.patient_id,), width="stretch")

    left, right = st.columns([3, 2])
    piv = risk_pivot(name)
    with left:
        st.markdown("### Clinic trend (last 30 days)")
        days = [d for d in piv.index if day - 30 <= d <= day]
        sub = piv.loc[days]
        ft = go.Figure()
        ft.add_trace(go.Bar(x=days, y=(sub >= THR).sum(axis=1), name="Patients in alert", marker_color="#dc2626"))
        ft.add_trace(go.Scatter(x=days, y=sub.mean(axis=1), name="Average risk %", yaxis="y2", line=dict(color="#2563eb", width=2.5)))
        ft.update_layout(height=320, margin=dict(t=10, b=30), legend=dict(orientation="h"),
                         yaxis=dict(title="Patients in alert"), yaxis2=dict(title="Average risk %", overlaying="y", side="right"),
                         xaxis_title="Day")
        st.plotly_chart(ft, width="stretch")
    with right:
        st.markdown("### New alerts this week")
        above = piv >= THR
        new = (above & ~above.shift(1, fill_value=False)).loc[max(piv.index.min(), day - 6):day]
        hits = sorted([(d, p) for d, row in new.iterrows() for p in row.index[row.values]], reverse=True)[:6]
        if not hits: st.info("No new alerts in the last 7 days.")
        for d, p in hits:
            rr = scored[(scored.patient_id == p) & (scored.day <= d)].tail(1)
            why = top_reasons(bundle, rr, k=1) if len(rr) else []
            st.markdown(f'''<div class="feed"><b>{p}</b> entered alert on day {d} ({piv.loc[d, p]:.0f}% risk)<br>
                <span style="color:#6b7280">{why[0][0] if why else ""}</span></div>''', unsafe_allow_html=True)

# ------------------------------------------------------------------------------------------------ patient
elif page == "Patient twin":
    prow = pats.loc[pid]
    rows = scored[scored.patient_id == pid].sort_values("day")
    cur = snap.set_index("patient_id").loc[pid]
    cur_idx = rows[rows.day <= day].index[-1]
    risk = float(cur.risk); alert = risk >= THR
    ev_p = events[events.patient_id == pid]

    c = st.columns(6)
    c[0].metric("7-day risk", f"{risk:.0f}%", f"band {cur.lo:.0f}-{cur.hi:.0f}%", delta_color="off")
    c[1].metric("Status", "ALERT" if alert else "Stable")
    c[2].metric("MELD-Na", f"{cur.meld_na:.0f}")
    c[3].metric("Child-Pugh", f"{cp_class(cur.child_pugh_pts)} ({int(cur.child_pugh_pts)} pts)")
    c[4].metric("Etiology", prow.etiology.upper())
    c[5].metric("Age / Sex", f"{prow.age} / {prow.sex}")

    # (2) risk timeline
    past = rows[rows.day <= day]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=past.day, y=past.hi, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=past.day, y=past.lo, fill="tonexty", fillcolor="rgba(220,60,60,0.15)",
                             line=dict(width=0), name="10-90% band"))
    fig.add_trace(go.Scatter(x=past.day, y=past.risk, name="7-day risk", line=dict(color="#c0392b", width=2.5)))
    if reveal:
        fut = rows[rows.day >= day]
        fig.add_trace(go.Scatter(x=fut.day, y=fut.risk, name="future (hidden)", line=dict(color="#999", dash="dot")))
    fig.add_hline(y=THR, line_dash="dash", line_color="#555", annotation_text="alert threshold")
    shown = ev_p if reveal else ev_p[ev_p.day <= day]
    for _, e in shown.iterrows():
        fig.add_vline(x=e.day, line_color="#2c3e50", line_width=1.5)
        fig.add_annotation(x=e.day, y=1, yref="paper", text=e.event_type, showarrow=False, textangle=-90,
                           font=dict(size=10), xanchor="right", yanchor="top")
    fig.add_vline(x=day, line_color="#2980b9", line_dash="dot")
    fig.update_layout(title="Risk timeline (probability of decompensation in the next 7 days)", height=340,
                      yaxis_title="Risk %", xaxis_title="Day", margin=dict(t=50, b=30), legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns([3, 2])
    # (3) wearable trends vs own baseline
    with left:
        d = daily[(daily.patient_id == pid) & (daily.day <= day) & (daily.day >= day - 60)]
        base = rows[rows.day.between(day - 60, day)].set_index("day")
        specs = [("steps", "Steps/day", "steps_base"), ("rest_hr", "Resting HR (bpm)", "rest_hr_base"),
                 ("hrv", "HRV RMSSD (ms)", "hrv_base"), ("night", "Night-time activity fraction", "night_base"),
                 ("mid", "Sleep midpoint (h after midnight)", "mid_base")]
        f2 = make_subplots(rows=len(specs), cols=1, shared_xaxes=True, vertical_spacing=0.04,
                           subplot_titles=[s[1] for s in specs])
        for i, (col, _, b) in enumerate(specs, 1):
            f2.add_trace(go.Scatter(x=d.day, y=d[col], mode="lines+markers", marker=dict(size=4),
                                    line=dict(width=1.5), showlegend=False), row=i, col=1)
            f2.add_trace(go.Scatter(x=base.index, y=base[b], line=dict(dash="dash", color="#888", width=2),
                                    showlegend=False), row=i, col=1)
        f2.update_layout(height=760, title="Wearables vs this patient's own twin baseline (dashed)", margin=dict(t=70, b=20))
        st.plotly_chart(f2, width="stretch")

    with right:
        # (4) labs, meds, twin state
        st.subheader("Latest labs & medication")
        labs = pd.DataFrame({"Test": ["Bilirubin (mg/dL)", "INR", "Creatinine (mg/dL)", "Sodium (mmol/L)", "Albumin (g/dL)",
                                      "Platelets (x10^3/uL)", "Ascites grade", "Encephalopathy grade"],
                             "Value": [f"{cur.bili:.1f}", f"{cur.inr:.2f}", f"{cur.creat:.2f}", f"{cur.sodium:.0f}",
                                       f"{cur.albumin:.1f}", f"{cur.platelets:.0f}", str(int(cur.ascites)), str(int(cur.he_grade))]})
        st.dataframe(labs, hide_index=True, width="stretch")
        st.caption(f"Labs last updated {int(cur.days_since_visit)} days ago.")
        st.write(f"**Medication:** beta-blocker {'yes' if prow.beta_blocker else 'no'}, diuretic {'yes' if prow.diuretic else 'no'}, "
                 f"rifaximin {'yes' if prow.rifaximin else 'no'}; lactulose adherence {prow.lactulose_adherence*100:.0f}%; "
                 f"alcohol: {'abstinent / n.a.' if prow.alcohol_abstinent else 'NOT abstinent'}.")
        st.write(f"**Setting:** {'rural' if prow.rural else 'urban'}, {prow.distance_km:.0f} km from specialist centre.")
        st.write(f"**Genetic marker:** PNPLA3 rs738409 risk alleles = {int(prow.pnpla3_risk)} (synthetic)")
        st.subheader("Twin state")
        s1, s2 = st.columns(2)
        s1.metric("Slow (severity)", f"{cur.twin_slow:.1f}", help="Smoothed MELD-Na trajectory")
        s2.metric("Fast (deviation)", f"{cur.twin_fast:.2f}", help="0 = at own baseline, 2 = far from baseline across signals")
        st.write(f"**Sleep-wake inversion score:** {cur.inversion:.2f}")

    # (5) alert + reasons
    st.subheader("Alert & reasons")
    X0 = scored.loc[[cur_idx]]
    reasons = top_reasons(bundle, X0, k=4)
    if alert:
        st.error(f"ALERT: estimated {risk:.0f}% chance of decompensation or unplanned admission within 7 days.")
        st.markdown("**Why (model explanation, SHAP):**")
        for s, v in reasons: st.markdown(f"- {s}")
        st.info("Suggested action: " + ACTIONS["default"])
    else:
        st.success(f"No alert. Estimated 7-day risk {risk:.0f}% (threshold {THR:.0f}%).")
        if reasons:
            st.markdown("**Largest contributors to current risk:**")
            for s, v in reasons[:3]: st.markdown(f"- {s}")
    if reveal:
        will = bool(cur.label)
        st.caption(f"Ground truth (synthetic): event within next 7 days = {'YES' if will else 'no'}.")

    # (6) what-if
    st.subheader("What-if simulator")
    st.caption("Model-based re-scoring of this patient's inputs. It reflects associations learned from SYNTHETIC data, not causal proof.")
    w1, w2, w3 = st.columns(3)
    adh = w1.slider("Lactulose adherence (%)", 10, 100, int(round(prow.lactulose_adherence * 100)), key=f"adh{pid}")
    steps = w2.slider("Daily steps change (%)", -30, 60, 0, key=f"st{pid}")
    sleep = w3.slider("Sleep-wake regularity improvement (%)", 0, 100, 0, key=f"sl{pid}")
    w4, w5 = st.columns(2)
    abst = w4.checkbox("Alcohol abstinent", bool(prow.alcohol_abstinent), key=f"ab{pid}")
    bb = w5.checkbox("On beta-blocker", bool(prow.beta_blocker), key=f"bb{pid}")
    row1 = apply_changes(X0.iloc[0], dict(lactulose_adherence=adh / 100, alcohol_abstinent=abst, beta_blocker=bb,
                                          steps_change_pct=steps, sleep_regularity_pct=sleep))
    r1, l1, h1 = predict_risk(bundle, pd.DataFrame([row1]))
    new = float(r1[0] * 100)
    k1, k2, k3 = st.columns(3)
    k1.metric("Current risk", f"{risk:.0f}%"); k2.metric("Scenario risk", f"{new:.0f}%", f"{new - risk:+.1f} pts", delta_color="inverse")
    k3.metric("Scenario band", f"{l1[0]*100:.0f}-{h1[0]*100:.0f}%")

# ------------------------------------------------------------------------------------------------ triage
elif page == "Clinic triage":
    st.subheader(f"Clinic triage list, day {day}")
    t = snap.copy()
    t["Child-Pugh"] = t.child_pugh_pts.map(cp_class)
    t["Status"] = np.where(t.risk >= THR, "ALERT", "stable")
    show = t[["patient_id", "etiology", "age", "meld_na", "Child-Pugh", "risk", "Status"]].rename(
        columns={"patient_id": "Patient", "etiology": "Etiology", "age": "Age", "meld_na": "MELD-Na", "risk": "7-day risk %"})
    if reveal: show["Event in next 7 d (truth)"] = t.label.map({1: "YES", 0: ""})
    st.dataframe(show.reset_index(drop=True), hide_index=True, width="stretch", height=520,
                 column_config={"7-day risk %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
                                "MELD-Na": st.column_config.NumberColumn(format="%.0f")})
    st.caption(f"{int((t.risk >= THR).sum())} of {len(t)} patients flagged today.")

# ------------------------------------------------------------------------------------------------ about
elif page == "About & limitations":
    st.markdown("""
**What this is.** A proof-of-concept digital twin for cirrhosis. Each patient has a persistent state that fuses a *slow* component
(smoothed MELD-Na) with a *fast* component (wearable deviation from the patient's own baseline) and predicts decompensation or
unplanned admission in the next 7 days.

**Limitations (please read).**
- All patients, wearables and events are **synthetic**. Wearable-to-event links come from assumptions in the generator, so results
  demonstrate the pipeline and method only. This is **not clinically validated**.
- Because we generate both signals and labels there is a **circularity risk**. Mitigations: events come from a hazard on a hidden state,
  35% of shocks are silent (no wearable prodrome), label noise, and a held-out cohort with different parameters (distribution shift).
- The what-if panel is model-based association, not causal evidence.
- MELD-Na uses the 2016 UNOS formula; MELD 3.0 exists. Verify clinical constants before any real-world use.
- Privacy: no real patient data is used (DPDP Act / HIPAA aware by design).
""")
