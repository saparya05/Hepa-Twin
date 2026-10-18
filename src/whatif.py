import numpy as np


def apply_changes(row, ch):
    r = row.copy()
    for k in ("lactulose_adherence", "alcohol_abstinent", "beta_blocker"):
        if k in ch: r[k] = float(ch[k])
    p = ch.get("steps_change_pct", 0) / 100.0
    if p:
        for c in ("steps_m3", "steps_m14"): r[c] = r[c] * (1 + p)
        for c in ("steps_dev3", "steps_dev7", "steps_twin"): r[c] = (1 + r[c]) * (1 + p) - 1
        r["steps_slope"] = r["steps_m3"] - r["steps_m14"]
    q = ch.get("sleep_regularity_pct", 0) / 100.0       # pull night/midpoint features back toward own baseline
    if q:
        for m in ("night", "mid"):
            base = r[f"{m}_base"]
            for c in (f"{m}_m3", f"{m}_m14"): r[c] = base + (r[c] - base) * (1 - q)
            for c in (f"{m}_dev3", f"{m}_dev7", f"{m}_twin"): r[c] = r[c] * (1 - q)
        r["inversion"] = r["inversion"] * (1 - q)
    return r
