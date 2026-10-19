"""Supporting analyses for the report.

  1. Adaptation index (deck gap 7): change in trading pace after a liquidity switch, minus the same policy's
     change at the same step in the fixed High condition (no switch). That separates reacting to the switch
     from a schedule that was front-loaded anyway.
  2. Value of observing the latent regime: oracle minus observable-only agents, stratified bootstrap.
  3. Idealised AC world vs data-calibrated LOB: method rankings and Kendall's tau (deck sub-question 4),
     and the cost of deploying AC-world-trained agents in the LOB.
  4. Sensitivity of the main comparisons to simulator variants: faster resilience, 10x latency, 3x fees.

    python3 experiments/extra_analysis.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

sys.path.insert(0, os.path.dirname(__file__))
from report import method_of, parse   # noqa: E402

rng = np.random.default_rng(1)
B = 2000


def load(d):
    df = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(d, "*.csv"))], ignore_index=True)
    df["method"] = df.policy.map(method_of)
    return df


def pace(df, cond, step, N=20):
    out = {}
    for meth, g in df[df.condition == cond].groupby("method"):
        P = np.array([[int(v) for v in s.split()] for s in g.path])
        sold = -np.diff(P, axis=1) / (P[:, 0].mean() / N)
        out[meth] = (sold[:, step - 3:step].mean(), sold[:, step:step + 3].mean(), P[:, step].mean() / P[:, 0].mean())
    return out


def runs_matrix(df, cond, meth, col="is_bps"):
    g = df[(df.condition == cond) & (df.method == meth)]
    return [x.sort_values("episode")[col].to_numpy() for _, x in g.groupby("policy")]


def boot_diff(A, Bm):
    """mean over runs of B minus mean over runs of A, stratified bootstrap (independent runs)."""
    vals = []
    for _ in range(B):
        a = [A[i][rng.integers(0, len(A[i]), len(A[i]))].mean() for i in rng.integers(0, len(A), len(A))]
        b = [Bm[i][rng.integers(0, len(Bm[i]), len(Bm[i]))].mean() for i in rng.integers(0, len(Bm), len(Bm))]
        vals.append(np.mean(b) - np.mean(a))
    return np.mean([x.mean() for x in Bm]) - np.mean([x.mean() for x in A]), np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def main():
    out = {}
    df = load("out/eval")

    # 1. adaptation index
    rows = []
    for cond, step in (("shock", 10), ("HML", 7)):
        sw, base = pace(df, cond, step), pace(df, "H", step)
        for meth in sw:
            b0, a0, inv = sw[meth]
            b1, a1, _ = base[meth]
            rows.append({"condition": cond, "method": meth, "pace_before": b0, "pace_after": a0, "inv_at_switch": inv,
                         "change": a0 - b0, "change_in_H": a1 - b1, "adaptation_index": (a0 - b0) - (a1 - b1)})
    # per-seed adaptation index for the RL methods (spread across training runs)
    for cond, step in (("shock", 10), ("HML", 7)):
        for meth in ("DDQN[mixed]", "PPO[mixed]"):
            vals = []
            for pol in df[(df.method == meth)].policy.unique():
                sub = df[df.policy == pol].copy()
                sub["method"] = pol
                sw, base = pace(sub, cond, step)[pol], pace(sub, "H", step)[pol]
                vals.append((sw[1] - sw[0]) - (base[1] - base[0]))
            r = next(x for x in rows if x["condition"] == cond and x["method"] == meth)
            r["per_seed"] = vals
            r["seed_se"] = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
    ad = pd.DataFrame(rows)
    out["adaptation"] = ad.to_dict("records")
    print(ad[ad.method.isin(["DDQN[mixed]", "PPO[mixed]"])][["condition", "method", "adaptation_index", "seed_se", "per_seed"]].to_string(index=False))
    print("== 1. adaptation index (pace change after the switch minus pace change at the same step with no switch)")
    print(ad[ad.method.isin(["TWAP", "VWAP", "POV", "AC", "AC-adaptive", "TWAP-LO", "DDQN[mixed]", "PPO[mixed]",
                             "DDQN[mixed+oracle]", "PPO[mixed+oracle]"])].round(2).to_string(index=False))

    # 2. oracle value
    rows = []
    for cond in ["L", "M", "H", "HML", "shock", "markov"]:
        for algo in ("DDQN", "PPO"):
            for col in ("is_bps", "objective"):
                d, lo, hi = boot_diff(runs_matrix(df, cond, f"{algo}[mixed]", col), runs_matrix(df, cond, f"{algo}[mixed+oracle]", col))
                rows.append({"condition": cond, "algo": algo, "metric": col, "oracle_minus_observable": d, "lo": lo, "hi": hi})
    orc = pd.DataFrame(rows)
    out["oracle"] = orc.to_dict("records")
    print("\n== 2. oracle (true regime observed) minus observable-only, mixed-trained agents; negative = information helps")
    print(orc[orc.metric == "objective"].round(2).to_string(index=False))

    # 3. AC world vs LOB rankings
    dac = load("out/eval_ac")
    rows = []
    for cond in ["L", "M", "H", "HML", "shock"]:
        a = dac[dac.condition == cond].groupby("method").objective.mean()
        a.index = [m.replace("|AC-world", "") for m in a.index]
        lob = df[df.condition == cond].groupby("method").objective.mean()
        common = [m for m in ["TWAP", "VWAP", "POV", "AC", "AC-adaptive", "DDQN[mixed]", "PPO[mixed]"] if m in a.index and m in lob.index]
        tau, p = kendalltau(a[common].rank(), lob[common].rank())
        def ranking(sr):
            """Best first; exact ties (identical schedules) joined with '='."""
            sr = sr.sort_values().round(6)
            groups = [" = ".join(g.index) for _, g in sr.groupby(sr, sort=True)]
            return " < ".join(groups)
        rows.append({"condition": cond, "kendall_tau_b": tau, "p": p,
                     "rank_ac_world": ranking(a[common]), "rank_lob": ranking(lob[common]),
                     **{f"acworld_{m}": a[m] for m in common}, **{f"lob_{m}": lob[m] for m in common}})
        # sim-to-real: AC-world agents deployed in the LOB vs LOB-trained
        for algo in ("DDQN", "PPO"):
            d, lo, hi = boot_diff(runs_matrix(df, cond, f"{algo}[mixed]", "objective"),
                                  runs_matrix(df, cond, f"{algo}[mixed|AC-world]", "objective"))
            rows[-1][f"{algo}_transfer_penalty"] = d
            rows[-1][f"{algo}_transfer_ci"] = (lo, hi)
    rk = pd.DataFrame(rows)
    out["ac_vs_lob"] = rk.to_dict("records")
    print("\n== 3. rankings by objective (best first), idealised AC world vs calibrated LOB")
    for r in rows:
        print(f"  {r['condition']:6s} Kendall tau-b = {r['kendall_tau_b']:+.2f} (p = {r['p']:.2f})\n     AC world: {r['rank_ac_world']}\n     LOB:      {r['rank_lob']}")
        print(f"     deploying AC-world agents in the LOB costs DDQN {r['DDQN_transfer_penalty']:+.2f} "
              f"[{r['DDQN_transfer_ci'][0]:+.2f}, {r['DDQN_transfer_ci'][1]:+.2f}], PPO {r['PPO_transfer_penalty']:+.2f} "
              f"[{r['PPO_transfer_ci'][0]:+.2f}, {r['PPO_transfer_ci'][1]:+.2f}] bps of objective")

    # 4. sensitivity
    print("\n== 4. sensitivity of the main comparisons (mixed-trained agents)")
    sens = {"base": "out/eval", "fast resilience": "out/eval_sens_fastres", "latency 0.5 s": "out/eval_sens_latency",
            "fees x3": "out/eval_sens_fees"}
    rows = []
    for name, d in sens.items():
        s = load(d)
        for cond in ("M", "shock"):
            t = s[s.condition == cond]
            tw = t[t.policy == "TWAP"].set_index("episode")
            row = {"variant": name, "condition": cond}
            for meth in ("TWAP", "AC", "POV", "DDQN[mixed]", "PPO[mixed]"):
                g = t[t.method == meth]
                row[f"{meth} IS"] = g.is_bps.mean()
                row[f"{meth} obj"] = g.objective.mean()
            for col, lab in (("is_bps", "IS"), ("objective", "obj")):
                dd, lo, hi = boot_diff(runs_matrix(t, cond, "DDQN[mixed]", col), runs_matrix(t, cond, "PPO[mixed]", col))
                row[f"PPO-DDQN {lab}"] = dd
                row[f"PPO-DDQN {lab} CI"] = f"[{lo:+.2f}, {hi:+.2f}]"
            for algo in ("DDQN", "PPO"):
                g = t[t.method == f"{algo}[mixed]"]
                diff = g.is_bps.to_numpy() - tw.is_bps.reindex(g.episode).to_numpy()
                row[f"{algo}-TWAP IS"] = diff.mean()
            rows.append(row)
    sv = pd.DataFrame(rows)
    out["sensitivity"] = sv.to_dict("records")
    cols = ["variant", "condition", "TWAP IS", "AC obj", "DDQN[mixed] IS", "PPO[mixed] IS", "DDQN-TWAP IS", "PPO-TWAP IS",
            "PPO-DDQN IS", "PPO-DDQN IS CI", "DDQN[mixed] obj", "PPO[mixed] obj", "PPO-DDQN obj", "PPO-DDQN obj CI"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(sv[cols].round(2).to_string(index=False))
    json.dump(out, open("out/report/extra.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
