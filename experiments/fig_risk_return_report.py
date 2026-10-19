"""Figure 5.4 of the report: mean shortfall against CVaR95 in settings B (Medium) and D2 (shock).

    python3 experiments/fig_risk_return_report.py      # reads out/report/methods.csv, writes out/report/fig5_4_risk_return.png
"""
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

M = pd.read_csv("out/report/methods.csv")
METHODS = [("TWAP", "TWAP", "#9aa3ad", "D"), ("TWAP-LO", "TWAP-LO", "#9aa3ad", "v"), ("VWAP", "VWAP", "#9aa3ad", "s"),
           ("POV", "POV", "#9aa3ad", "^"), ("AC", "Almgren–Chriss", "#e8692d", "s"),
           ("AC-adaptive", "AC-adaptive", "#e8692d", "D"), ("DDQN[mixed]", "DDQN", "#2a7ae2", "o"),
           ("PPO[mixed]", "PPO", "#1fae7a", "o")]
OFFSETS = {("shock", "VWAP"): (6, 4), ("shock", "TWAP"): (6, -12), ("M", "VWAP"): (6, -12), ("M", "TWAP"): (6, 4),
           ("M", "AC-adaptive"): (6, -12), ("shock", "AC-adaptive"): (6, -12)}
fig, axes = plt.subplots(1, 2, figsize=(10.1, 4.0))
for ax, (cond, title) in zip(axes, (("M", "B: Medium"), ("shock", "D2: H→L shock"))):
    d = M[M.condition == cond].set_index("method")
    for key, label, col, mk in METHODS:
        r = d.loc[key]
        ax.scatter(r.cvar95, r["is"], color=col, marker=mk, s=60 if "[" in key else 45, zorder=3)
        ax.annotate(label, (r.cvar95, r["is"]), xytext=OFFSETS.get((cond, key), (6, 4)), textcoords="offset points",
                    fontsize=9)
    ax.set_title(title, loc="left", fontsize=12)
    ax.set_xlabel("CVaR95 of shortfall (bps), tail risk")
    ax.grid(color="#e6e6e6")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("Mean shortfall (bps)")
fig.tight_layout()
fig.savefig("out/report/fig5_4_risk_return.png", dpi=176)
print("out/report/fig5_4_risk_return.png")
