"""Simulator realism checks (Vyetrenko et al., 2020 style): untargeted statistics and emergent impact.

    python3 -m liqexec.validate_sim out/calib/regimes.json data/lobster/AAPL_..._message_10.csv --out out/validation

1. Untargeted distributions per regime, sim vs data: spread quantiles, 10-second return quantiles and
   excess kurtosis, market-order volume per minute.
2. Metaorder impact (the deck's Delta P ~ |v|^delta): sell X shares by TWAP over 10 minutes for several X;
   peak impact = E[mid at the end | agent] - E[mid at the end | no agent], with common random numbers.
   Fit delta on log X. Delta < 1 means concave impact, and it is emergent, not imposed.
3. Decay: the same impact measured 1, 5 and 15 minutes after the metaorder ends. Impact that falls back
   toward a lower level is transient; what remains is permanent.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from .calibrate import load
from .lobsim import Market


def data_stats(msg_path, windows_csv, skip_open=300.0, W=300.0):
    m, ap, asz, bp, bsz = load(msg_path)
    t = m.t.to_numpy()
    w = pd.read_csv(windows_csv)
    reg_of_w = dict(zip(w.w, w.regime))
    out = {}
    grid = np.arange(t[0] + skip_open, t[-1], 1.0)
    idx = np.searchsorted(t, grid, side="right") - 1
    mid = (ap[idx, 0] + bp[idx, 0]) / 2.0
    spread = ap[idx, 0] - bp[idx, 0]
    gw = ((grid - t[0] - skip_open) // W).astype(int)
    reg = np.array([reg_of_w.get(x, "?") for x in gw])
    for name in ("L", "M", "H"):
        s = reg == name
        r10 = []
        for wi in np.unique(gw[s]):                      # 10 s returns inside each window
            mm = mid[gw == wi]
            r10 += list(mm[10::10] - mm[:-10:10])
        r10 = np.array(r10)
        out[name] = {"spread_q": np.percentile(spread[s], [10, 25, 50, 75, 90]).tolist(),
                     "r10_q": np.percentile(r10, [1, 5, 25, 50, 75, 95, 99]).tolist(),
                     "r10_kurt": float(pd.Series(r10).kurt())}
    return out


def sim_stats(m, r, seeds=range(200, 210), T=1800):
    sp, r10 = [], []
    for s in seeds:
        m.reset(r, seed=s)
        mids = []
        for _ in range(T):
            m.run_until(m.t + 1.0)
            sp.append(m.best_ask() - m.best_bid())
            mids.append(m.mid())
        mids = np.array(mids)
        r10 += list(mids[10::10] - mids[:-10:10])
    r10 = np.array(r10)
    return {"spread_q": np.percentile(sp, [10, 25, 50, 75, 90]).tolist(),
            "r10_q": np.percentile(r10, [1, 5, 25, 50, 75, 95, 99]).tolist(), "r10_kurt": float(pd.Series(r10).kurt())}


def metaorder(m, r, X, seeds, T=600.0, n=20, after=(60.0, 300.0, 900.0)):
    """Mid displacement (ticks, agent minus no-agent, same seeds) at the end of a TWAP sell and later."""
    res = np.zeros((len(seeds), 1 + len(after)))
    for i, s in enumerate(seeds):
        for with_agent in (True, False):
            m.reset(r, seed=s)
            t0, m0 = m.t, m.mid()
            for k in range(n):
                if with_agent:
                    m.agent_market(X // n)
                m.run_until(t0 + (k + 1) * T / n)
            vals = [m.mid() - m0]
            for h in after:
                m.run_until(t0 + T + h)
                vals.append(m.mid() - m0)
            res[i] += np.array(vals) * (1 if with_agent else -1)
    return -res          # positive = price pushed down by the seller


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("calib")
    ap.add_argument("messages")
    ap.add_argument("--out", default="out/validation")
    ap.add_argument("--seeds", type=int, default=150)
    ap.add_argument("--xs", type=int, nargs="+", default=[1000, 2000, 4000, 8000, 16000, 32000])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    m = Market(args.calib)
    windows_csv = os.path.join(os.path.dirname(args.calib), "windows.csv")
    ds = data_stats(args.messages, windows_csv)
    report = {"distributions": {}, "metaorder": {}}
    print("== untargeted distributions (data vs sim)")
    for r, name in enumerate(m.regimes):
        ss = sim_stats(m, r)
        report["distributions"][name] = {"data": ds[name], "sim": ss}
        print(f"{name}: spread p10/25/50/75/90  data {np.round(ds[name]['spread_q']).astype(int)}  sim {np.round(ss['spread_q']).astype(int)}")
        print(f"   10s return p1/5/25/50/75/95/99  data {np.round(ds[name]['r10_q'], 1)}  sim {np.round(ss['r10_q'], 1)}")
        print(f"   10s return excess kurtosis  data {ds[name]['r10_kurt']:.1f}  sim {ss['r10_kurt']:.1f}")

    print("\n== metaorder impact: TWAP sell over 10 min, 20 slices (ticks; mean +- s.e.)")
    Xs = args.xs
    rng = np.random.default_rng(0)
    for r, name in enumerate(m.regimes):
        rows, per = [], []
        for X in Xs:
            res = metaorder(m, r, X, seeds=range(1000, 1000 + args.seeds))
            per.append(res)
            mu, se = res.mean(0), res.std(0, ddof=1) / np.sqrt(len(res))
            rows.append((X, mu.tolist(), se.tolist()))
            print(f"{name} X={X:6d}: end {mu[0]:6.1f}+-{se[0]:.1f}   +1m {mu[1]:6.1f}   +5m {mu[2]:6.1f}   +15m {mu[3]:6.1f}",
                  flush=True)
        x = np.log(np.array(Xs, float))
        per = np.stack(per)                                    # [X, seed, horizon]
        fit = lambda P: np.polyfit(x, np.log(np.maximum(P[:, :, 0].mean(1), 1e-3)), 1)[0]
        delta = fit(per)
        boots = [fit(per[:, rng.integers(0, per.shape[1], per.shape[1])]) for _ in range(1000)]   # resample seeds jointly
        lo, hi = np.percentile(boots, [2.5, 97.5])
        peak, after5 = per[-1, :, 0].mean(), per[-1, :, 2].mean()
        rb = [1 - per[-1, i, 2].mean() / per[-1, i, 0].mean() for i in
              (rng.integers(0, per.shape[1], per.shape[1]) for _ in range(1000))]
        report["metaorder"][name] = {"X": Xs, "rows": rows, "delta": float(delta), "delta_ci": [float(lo), float(hi)],
                                     "reversion_5min_largest": float(1 - after5 / peak),
                                     "reversion_ci": [float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))]}
        print(f"   {name}: impact ~ X^{delta:.2f}  (95% CI {lo:.2f} to {hi:.2f});  share of peak impact reverted 5 min after"
              f" the largest order: {1 - after5 / peak:.0%} (95% CI {np.percentile(rb, 2.5):.0%} to {np.percentile(rb, 97.5):.0%})")
    json.dump(report, open(os.path.join(args.out, "validation.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
