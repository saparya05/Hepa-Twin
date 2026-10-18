import json, os, time
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from . import data_gen
from .features import build_features, ALL_COLS, EHR_COLS, WEARABLE_COLS
from .modeling import fit_lgbm, predict_risk
from .evaluate import row_metrics, events_by_patient, alert_metrics, choose_threshold

TARGET_FA = 1.0     # false alerts per patient-month allowed when picking the alert threshold
N_ENS = 6


def load(name, d="data"):
    pat, daily, ev = (pd.read_csv(f"{d}/{name}_{k}.csv") for k in ("patients", "daily", "events"))
    return pat, daily, ev, build_features(pat, daily)


def main():
    t0 = time.time()
    if not os.path.exists("data/train_daily.csv"):
        data_gen.main("data")
    os.makedirs("models", exist_ok=True); os.makedirs("results", exist_ok=True)
    print("Building features (replaying each patient through the twin)...")
    _, _, ev_tr, Ftr = load("train"); _, _, ev_te, Fte = load("test"); _, _, ev_sh, Fsh = load("shift")
    print(f"  rows: train {len(Ftr)}  test {len(Fte)}  shift {len(Fsh)} | window-positive rate: "
          f"{Ftr.label.mean():.3f} / {Fte.label.mean():.3f} / {Fsh.label.mean():.3f}")

    rng = np.random.default_rng(0)
    ids = np.array(sorted(Ftr.patient_id.unique())); rng.shuffle(ids)
    val_ids = set(ids[: int(0.2 * len(ids))])
    is_val = Ftr.patient_id.isin(val_ids)
    A, V = Ftr[~is_val].reset_index(drop=True), Ftr[is_val].reset_index(drop=True)   # split by PATIENT

    # ---- ablation -------------------------------------------------------------------------------
    res = {"ablation": {}}
    for name, cols in [("Wearable-only", WEARABLE_COLS), ("EHR-only", EHR_COLS), ("Fused (EHR + wearable)", ALL_COLS)]:
        m = fit_lgbm(A[cols], A.label)
        res["ablation"][name] = {c: row_metrics(F.label, m.predict_proba(F[cols])[:, 1])
                                 for c, F in (("test", Fte), ("shift", Fsh))}
        print(f"  {name:24s} AUROC test {res['ablation'][name]['test']['auroc']:.3f} | shift {res['ablation'][name]['shift']['auroc']:.3f}")
    lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=500, C=0.3))
    lr.fit(A[ALL_COLS], A.label)
    res["ablation"]["Logistic regression (fused)"] = {c: row_metrics(F.label, lr.predict_proba(F[ALL_COLS])[:, 1])
                                                     for c, F in (("test", Fte), ("shift", Fsh))}

    # ---- final fused bootstrap ensemble + isotonic calibration ----------------------------------
    idx = A.groupby("patient_id").indices
    pids = np.array(list(idx))
    models = []
    for i in range(N_ENS):
        pick = np.random.default_rng(100 + i).choice(pids, len(pids), replace=True)
        rows = np.concatenate([idx[p] for p in pick])
        models.append(fit_lgbm(A.iloc[rows][ALL_COLS], A.iloc[rows].label, seed=i))
    bundle = dict(models=models, feature_cols=ALL_COLS, iso=None, threshold=None, target_fa=TARGET_FA)
    raw_val = np.mean([m.predict_proba(V[ALL_COLS])[:, 1] for m in models], axis=0)
    bundle["iso"] = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw_val, V.label)

    scored = {}
    for name, F in (("val", V), ("test", Fte), ("shift", Fsh)):
        r, lo, hi = predict_risk(bundle, F)
        scored[name] = F.assign(risk=r * 100, lo=lo * 100, hi=hi * 100)
    ev_val = events_by_patient(ev_tr[ev_tr.patient_id.isin(val_ids)])
    # alert logic operates on risk in percent
    thr = choose_threshold(scored["val"], ev_val, TARGET_FA)
    bundle["threshold"] = thr / 100.0
    res["threshold_pct"], res["target_false_alerts_per_patient_month"] = thr, TARGET_FA
    res["final"] = {}
    for name, F, ev in (("test", scored["test"], ev_te), ("shift", scored["shift"], ev_sh)):
        m = row_metrics(F.label, F.risk / 100, calibrated=True)
        m["alerts"] = alert_metrics(F, events_by_patient(ev), thr)
        res["final"][name] = m
        a = m["alerts"]
        print(f"  FINAL {name:5s} AUROC {m['auroc']:.3f} AUPRC {m['auprc']:.3f} Brier {m['brier']:.3f} | "
              f"detect {a['event_detection_rate']:.2f} lead {a['median_lead_time_days']} d | FA/pt-month {a['false_alerts_per_patient_month']:.2f}")

    joblib.dump(bundle, "models/hepatwin.joblib")
    scored["test"].to_pickle("data/scored_test.pkl"); scored["shift"].to_pickle("data/scored_shift.pkl")
    json.dump(res, open("results/results.json", "w"), indent=2)

    # ---- report artefacts -----------------------------------------------------------------------
    L = ["| Model | AUROC (test) | AUPRC (test) | AUROC (shift) | AUPRC (shift) |", "|---|---|---|---|---|"]
    for k, v in res["ablation"].items():
        L.append(f"| {k} | {v['test']['auroc']:.3f} | {v['test']['auprc']:.3f} | {v['shift']['auroc']:.3f} | {v['shift']['auprc']:.3f} |")
    L += ["", "| Fused ensemble (calibrated) | Test | Shift |", "|---|---|---|"]
    f = res["final"]
    def row(lbl, fn): return f"| {lbl} | {fn(f['test'])} | {fn(f['shift'])} |"
    L += [row("AUROC", lambda x: f"{x['auroc']:.3f}"), row("AUPRC", lambda x: f"{x['auprc']:.3f}"),
          row("Brier score", lambda x: f"{x['brier']:.3f}"), row("Sensitivity @ 90% specificity", lambda x: f"{x['sens_at_90spec']:.2f}"),
          row("Events detected (alert in prior 7 d)", lambda x: f"{x['alerts']['event_detection_rate']*100:.0f}%"),
          row("Median lead time (days)", lambda x: str(x['alerts']['median_lead_time_days'])),
          row("False alerts / patient-month", lambda x: f"{x['alerts']['false_alerts_per_patient_month']:.2f}")]
    open("results/ablation.md", "w").write("\n".join(L) + f"\n\nAlert threshold: {thr:.1f}% risk (chosen on validation patients for <= {TARGET_FA} false alerts per patient-month).\n")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for nm, F in (("test", scored["test"]), ("shift", scored["shift"])):
        fr, mp = calibration_curve(F.label, F.risk / 100, n_bins=8, strategy="quantile")
        ax[0].plot(mp, fr, "o-", label=nm)
    ax[0].plot([0, 1], [0, 1], "k--", lw=1); ax[0].set(xlabel="Predicted risk", ylabel="Observed frequency", title="Calibration"); ax[0].legend()
    names = list(res["ablation"]); x = np.arange(len(names))
    ax[1].bar(x - 0.2, [res["ablation"][n]["test"]["auroc"] for n in names], 0.4, label="test")
    ax[1].bar(x + 0.2, [res["ablation"][n]["shift"]["auroc"] for n in names], 0.4, label="shift")
    ax[1].set_xticks(x); ax[1].set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=7); ax[1].set(ylim=(0.5, 1), title="Ablation (AUROC)"); ax[1].legend()
    plt.tight_layout(); plt.savefig("results/calibration_ablation.png", dpi=150)
    print(f"Done in {time.time()-t0:.0f}s. See results/ablation.md")


if __name__ == "__main__":
    main()
