"""Draws the architecture diagram -> docs/architecture.pdf / .png.  Run: python docs/make_architecture.py"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(13, 7)); ax.set_xlim(0, 13); ax.set_ylim(0, 7); ax.axis("off")
def box(x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12", fc=fc, ec="#34495e", lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)
def arrow(x1, y1, x2, y2): ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", lw=1.5, color="#34495e"))

ax.text(6.5, 6.75, "Hepa-Twin architecture (all data synthetic)", ha="center", fontsize=14, weight="bold")
box(0.2, 4.6, 2.6, 1.3, "STATIC STREAM\nSynthetic EHR\nlabs, MELD-Na, Child-Pugh,\nmeds, etiology, history", "#d6eaf8")
box(0.2, 1.2, 2.6, 1.6, "DYNAMIC STREAM\nSynthetic wearables (daily)\nsteps, resting HR, HRV, sleep,\nnight activity, sleep midpoint", "#d5f5e3")
box(0.2, 3.1, 2.6, 1.1, "Latent-state generator\nreserve + shocks -> hazard\n(35% silent shocks)", "#fdebd0")
box(3.4, 3.0, 2.8, 2.4, "DIGITAL TWIN STATE\n(persistent per patient)\nSLOW: smoothed MELD-Na\nFAST: deviation from own\nadaptive baseline\nsleep-wake inversion score", "#fcf3cf")
box(6.8, 3.0, 2.6, 2.4, "Fused model\nLightGBM bootstrap ensemble\n+ isotonic calibration\n=> 7-day risk, 10-90% band\nalert threshold (FA/pt-month)", "#f5b7b1")
box(10.0, 4.7, 2.8, 1.2, "Explainability\nSHAP -> plain-language reasons", "#e8daef")
box(10.0, 3.0, 2.8, 1.3, "What-if simulator\nadherence, steps, sleep regularity,\nabstinence, beta-blocker", "#e8daef")
box(10.0, 0.9, 2.8, 1.7, "Doctor dashboard (Streamlit)\nrisk timeline, wearables vs baseline,\nlabs, alert + reasons, what-if,\nclinic triage list", "#d4e6f1")
box(3.4, 0.9, 6.0, 1.2, "Evaluation: ablation (wearable-only / EHR-only / fused), calibration, lead time,\nfalse alerts per patient-month, held-out + distribution-shift cohorts", "#eaecee")
for a in [(2.8, 5.2, 3.4, 4.7), (2.8, 2.0, 3.4, 3.4), (1.5, 4.6, 1.5, 4.2), (1.5, 3.1, 1.5, 2.8), (6.2, 4.2, 6.8, 4.2),
          (9.4, 4.7, 10.0, 5.2), (9.4, 3.9, 10.0, 3.7), (11.4, 4.7, 11.4, 4.3), (11.4, 3.0, 11.4, 2.6), (8.1, 3.0, 8.1, 2.1)]: arrow(*a)
ax.text(6.5, 0.35, "Decision support only. Synthetic data. Not for clinical use.", ha="center", fontsize=9, style="italic", color="#c0392b")
plt.savefig("docs/architecture.pdf", bbox_inches="tight"); plt.savefig("docs/architecture.png", dpi=150, bbox_inches="tight")
