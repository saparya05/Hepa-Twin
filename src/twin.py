import numpy as np
import pandas as pd

# metric -> (kind, gate).  'rel' = relative deviation x/base-1, 'abs' = x-base
METRICS = {"steps": ("rel", 0.35), "rest_hr": ("rel", 0.10), "hrv": ("rel", 0.35),
           "sleep_dur": ("rel", 0.25), "sleep_eff": ("rel", 0.10),
           "mid": ("abs", 1.2), "night": ("abs", 0.06)}


class PatientTwin:
    def __init__(self, alpha_base=0.03, alpha_dev=0.35, alpha_slow=0.25, warmup=7):
        self.ab, self.ad, self.asl, self.warm = alpha_base, alpha_dev, alpha_slow, warmup
        self.base = {m: np.nan for m in METRICS}
        self.n = {m: 0 for m in METRICS}
        self.dev = {m: 0.0 for m in METRICS}
        self.slow = np.nan

    def _dev(self, m, x):
        return x / self.base[m] - 1 if METRICS[m][0] == "rel" else x - self.base[m]

    def update(self, obs, meld=None):
        """Feed one day of data. Missing values leave the state unchanged."""
        for m, (kind, gate) in METRICS.items():
            x = obs.get(m, np.nan)
            if x is None or np.isnan(x):
                continue
            if self.n[m] < self.warm:                      # warm-up: running mean
                self.n[m] += 1
                self.base[m] = x if self.n[m] == 1 else self.base[m] + (x - self.base[m]) / self.n[m]
            d = self._dev(m, x)
            if self.n[m] >= self.warm and abs(d) < gate:   # gated baseline adaptation
                self.base[m] += self.ab * (x - self.base[m])
            self.dev[m] = self.ad * d + (1 - self.ad) * self.dev[m]
        if meld is not None and not np.isnan(meld):
            self.slow = meld if np.isnan(self.slow) else self.slow + self.asl * (meld - self.slow)
        return self.state()

    def state(self):
        s = {}
        for m in METRICS:
            s[f"base_{m}"], s[f"twin_dev_{m}"] = self.base[m], self.dev[m]
        s["twin_fast"] = float(np.mean([min(abs(self.dev[m]) / METRICS[m][1], 2.0) for m in METRICS]))
        s["twin_slow"] = self.slow
        return s


def run_twin(g: pd.DataFrame) -> pd.DataFrame:
    """Replay one patient's daily records through the twin (day by day, causal)."""
    tw = PatientTwin()
    cols = list(METRICS)
    X, M = g[cols].to_numpy(dtype=float), g["meld_na"].to_numpy(dtype=float)
    rows = [tw.update(dict(zip(cols, X[i])), M[i]) for i in range(len(g))]
    return pd.DataFrame(rows, index=g.index)
