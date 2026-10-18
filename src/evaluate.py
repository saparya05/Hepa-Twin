import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve


def sens_at_spec(y, p, spec=0.90):
    fpr, tpr, _ = roc_curve(y, p)
    return float(tpr[fpr <= 1 - spec].max())


def row_metrics(y, p, calibrated=False):
    out = dict(auroc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
               sens_at_90spec=sens_at_spec(y, p), prevalence=float(np.mean(y)))
    if calibrated:
        out["brier"] = float(brier_score_loss(y, p))
    return out


def events_by_patient(ev_df):
    return {pid: g["day"].to_numpy() for pid, g in ev_df.groupby("patient_id")}


def alert_days(days, risk, thr, cooldown=7):
    out, last = [], -10 ** 6
    for d, r in zip(days, risk):
        if r >= thr and d - last >= cooldown:
            out.append(d); last = d
    return np.array(out, dtype=int)


def alert_metrics(df, ev_map, thr, horizon=7, cooldown=7):
    """Alert fires when risk >= thr (re-fires at most once per `cooldown` days).
    Detected event = alert in the `horizon` days before it. False alert = alert with no event in next `horizon` days."""
    n_alert = n_false = n_events = n_det = 0
    leads = []
    for pid, g in df.groupby("patient_id"):
        g = g.sort_values("day")
        days, risk = g["day"].to_numpy(), g["risk"].to_numpy()
        al = alert_days(days, risk, thr, cooldown)
        evd = ev_map.get(pid, np.array([], dtype=int))
        n_alert += len(al)
        for a in al:
            if not ((evd > a) & (evd <= a + horizon)).any():
                n_false += 1
        for ed in evd:
            if not ((days >= ed - horizon) & (days <= ed - 1)).any():
                continue                                   # event not coverable by a scored day
            n_events += 1
            w = al[(al >= ed - horizon) & (al <= ed - 1)]
            if len(w):
                n_det += 1; leads.append(int(ed - w.min()))
    patient_months = len(df) / 30.0
    return dict(threshold=float(thr), events=n_events, detected=n_det,
                event_detection_rate=n_det / max(n_events, 1),
                median_lead_time_days=float(np.median(leads)) if leads else None,
                alerts=n_alert, false_alerts=n_false,
                false_alerts_per_patient_month=n_false / patient_months)


def choose_threshold(df, ev_map, target_fa=1.0, n_grid=60):
    """Smallest risk threshold whose false alerts per patient-month on `df` is <= target.
    NOTE: per-patient-month = false alerts / (scored patient-days / 30)."""
    qs = np.unique(np.quantile(df["risk"], np.linspace(0.5, 0.999, n_grid)))
    for thr in qs:
        if alert_metrics(df, ev_map, thr)["false_alerts_per_patient_month"] <= target_fa:
            return float(thr)
    return float(qs[-1])
