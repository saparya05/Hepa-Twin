import numpy as np
import lightgbm as lgb

PARAMS = dict(n_estimators=250, learning_rate=0.04, num_leaves=15, min_child_samples=40, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0, verbose=-1, n_jobs=2)


def fit_lgbm(X, y, seed=0):
    m = lgb.LGBMClassifier(**PARAMS, random_state=seed)
    m.fit(X, y)
    return m


def predict_risk(bundle, X):
    """Returns calibrated risk (0-1) plus a 10th-90th percentile band across ensemble members."""
    Xf = X[bundle["feature_cols"]]
    P = np.column_stack([m.predict_proba(Xf)[:, 1] for m in bundle["models"]])
    iso = bundle["iso"]
    risk = iso.predict(P.mean(1))
    members = np.column_stack([iso.predict(P[:, i]) for i in range(P.shape[1])])
    lo, hi = np.percentile(members, [10, 90], axis=1)
    return risk, np.minimum(lo, risk), np.maximum(hi, risk)
