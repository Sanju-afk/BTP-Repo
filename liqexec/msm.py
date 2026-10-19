"""Method of simulated moments: second and third calibration stages.

Stage "static" (per regime). The queue-reactive intensities from calibrate.py are maximum-likelihood
estimates per (level, queue size). Simulated from those rates alone, the book has a narrower spread than
the data (in AAPL by 22-35%) and, in the Low and Medium regimes, more depth (by 35% and 20%), because conditioning on queue size averages fleeting
orders (inserted inside the spread and cancelled within milliseconds) with ordinary queues. This stage
scales four groups of rates:
  m_ins (inserts inside the spread), m_cx1 (cancellations at level 1), m_cxd (cancellations at levels 2..K),
  m_lo (limit orders joining existing levels)
to match mean spread, mean depth over 5 levels, mean level-1 queue and 1-second mid volatility.

Stage "dyn" (shared by all regimes). Fits the order-flow parameters of lobsim._run: the share pi_m of
market orders that belong to background metaorders, their Pareto length exponent alpha, the tilt kappa
toward the fundamental price, the fundamental's volatility sigma_f, the permanent share theta_p, and the
post-trade quote-revision probabilities w_hit and w_opp. (The Hawkes branching ratio is not fitted
here; it is a direct maximum-likelihood estimate.) The targets are the data's market-order response function R(10 s),
the ratios R(120)/R(10) and R(300)/R(10) (how much of the impact reverts), the sign autocorrelation at
lags 1, 5 and 10, and the variance ratios VR(60) and VR(300).

Both stages use Nelder-Mead on log parameters with common random numbers (fixed seeds), with multi-start.
`--stage all` alternates static, dyn, static, because the stages interact through volatility.

    python3 -m liqexec.msm out/calib/regimes.json --stage all
"""
import argparse
import json

import numpy as np
from scipy.optimize import minimize

from .lobsim import Market

MOMENTS = ("spread_ticks", "depth5", "q1", "vol_ticks_per_sqrt_s")
DYN = ("R10", "rev120", "rev300", "acf1", "acf5", "acf10", "VR60", "VR300")


def sim_moments(m, r, seeds=(11, 12, 13, 14), T=1800):
    sp, d5, q1, vol, mov = [], [], [], [], []
    for s in seeds:
        m.reset(r, seed=s)
        mids = []
        for _ in range(T):
            m.run_until(m.t + 1.0)
            sp.append(m.best_ask() - m.best_bid())
            d5.append((m.depth(0, 5) + m.depth(1, 5)) / 2)
            q1.append((m.depth(0, 1) + m.depth(1, 1)) / 2)
            mids.append(m.mid())
        b, s_, _, _ = m.take_flow()
        vol.append(np.std(np.diff(mids)))
        mov.append((b + s_) / T)
    return {"spread_ticks": np.mean(sp), "depth5": np.mean(d5), "q1": np.mean(q1),
            "vol_ticks_per_sqrt_s": np.sqrt(np.mean(np.square(vol))), "mo_volume_per_s": np.mean(mov)}


DYN_SEEDS = tuple(range(21, 31))
SIM_HOURS = 3 * len(DYN_SEEDS)


def dyn_moments(m, seeds=DYN_SEEDS, T=3600):
    """Response function, sign autocorrelation and variance ratios pooled over regimes (equal weight)."""
    R = {10: [], 120: [], 300: []}
    acf, v1, v60, v300 = {1: [], 5: [], 10: []}, [], [], []
    for r in range(len(m.regimes)):
        for s in seeds:
            m.reset(r, seed=s)
            g0 = m.fs[2]                                   # time of mids[0]
            m.run_until(m.t + T)
            mids = m.mids[:int(m.st[9])].copy()
            log = m.molog[:int(m.st[14])].copy()
            idx0 = np.ceil(log[:, 0] - g0).astype(int)
            for h in R:
                j = idx0 + h
                ok = j < len(mids)
                R[h] += list(log[ok, 1] * (mids[j[ok]] - log[ok, 2]))
            for l in acf:
                acf[l].append(np.corrcoef(log[:-l, 1], log[l:, 1])[0, 1])
            d1 = np.diff(mids)
            v1.append(np.var(d1))
            v60.append(np.var(mids[60:] - mids[:-60]))
            v300.append(np.var(mids[300:] - mids[:-300]))
    R10 = np.mean(R[10])
    return {"R10": R10, "rev120": np.mean(R[120]) / R10, "rev300": np.mean(R[300]) / R10,
            "acf1": np.mean(acf[1]), "acf5": np.mean(acf[5]), "acf10": np.mean(acf[10]),
            "VR60": np.mean(v60) / (60 * np.mean(v1)), "VR300": np.mean(v300) / (300 * np.mean(v1))}


def data_dyn_targets(p):
    d = p["data_dyn"]
    tg = {"R10": d["R10"], "rev120": d["R120"] / d["R10"], "rev300": d["R300"] / d["R10"],
          "acf1": d["acf1"], "acf5": d["acf5"], "acf10": d["acf10"], "VR60": d["VR60"], "VR300": d["VR300"]}
    # standard error of (sim - data): data block-bootstrap s.e., inflated for simulation noise
    # (simulated hours vs data hours)
    data_hours = p["window_s"] * sum(v["data"]["windows"] for v in p["per_regime"].values()) / 3600
    tg["se"] = {k: v * np.sqrt(1 + data_hours / SIM_HOURS) for k, v in d["se"].items()}
    return tg


def dyn_loss(mo, tg):
    """Optimally weighted (diagonal) MSM criterion: sum of squared t-statistics."""
    return sum(((mo[k] - tg[k]) / tg["se"][k]) ** 2 for k in DYN)


def apply_multipliers(m, r, mult, raw):
    q = m.P[r]
    q["lam_ins"] = raw["lam_ins"] * mult[0]
    q["lam_cx"] = raw["lam_cx"].copy()
    q["lam_cx"][0] *= mult[1]
    q["lam_cx"][1:] *= mult[2]
    q["lam_lo"] = raw["lam_lo"] * mult[3]


def stage_static(p, iters):
    prev = p.pop("msm", {})
    m = Market(p)                         # raw rates (no multipliers), current dyn parameters
    out = {}
    for r, name in enumerate(m.regimes):
        data = p["per_regime"][name]["data"]
        raw = {k: m.P[r][k].copy() for k in ("lam_ins", "lam_cx", "lam_lo")}
        target = np.log([data[k] for k in MOMENTS])

        def loss(logm):
            apply_multipliers(m, r, np.exp(logm), raw)
            mo = sim_moments(m, r)
            return float(np.sum((np.log([mo[k] for k in MOMENTS]) - target) ** 2))

        before = sim_moments(m, r)
        starts = [np.log([v["m_ins"], v["m_cx1"], v["m_cxd"], v["m_lo"]]) for v in ([prev[name]] if name in prev else [])]
        starts += [np.zeros(4)] + [np.log([v["m_ins"], v["m_cx1"], v["m_cxd"], v["m_lo"]]) for v in out.values()]
        best = None
        for x0 in starts:                          # multi-start: Nelder-Mead is local
            simplex = x0 + np.array([[0, 0, 0, 0], [0.4, 0, 0, 0], [0, 0.4, 0, 0], [0, 0, 0.4, 0], [0, 0, 0, -0.4]])
            res = minimize(loss, x0, method="Nelder-Mead",
                           options={"maxfev": iters, "xatol": 0.02, "fatol": 1e-4, "initial_simplex": simplex})
            if best is None or res.fun < best.fun:
                best = res
            if best.fun < 0.01:
                break
        mult = np.exp(best.x)
        apply_multipliers(m, r, mult, raw)
        after = sim_moments(m, r, seeds=(101, 102, 103, 104))          # fresh seeds: out-of-sample check
        out[name] = {"m_ins": mult[0], "m_cx1": mult[1], "m_cxd": mult[2], "m_lo": mult[3], "loss": best.fun,
                     "before": before, "after_fresh_seeds": after, "data": {k: data[k] for k in MOMENTS + ("mo_volume_per_s",)}}
        print(f"{name}: m_ins {mult[0]:.2f}  m_cx1 {mult[1]:.2f}  m_cxd {mult[2]:.2f}  m_lo {mult[3]:.2f}  (loss {best.fun:.4f})")
        for k in MOMENTS + ("mo_volume_per_s",):
            print(f"    {k:22s} data {data[k]:8.2f}   raw rates {before[k]:8.2f}   fitted (fresh seeds) {after[k]:8.2f}")
    p["msm"] = out


def stage_dyn(p, iters):
    tg = data_dyn_targets(p)
    m = Market(p)
    d0 = p.get("dyn", {})

    sig = lambda v: 1 / (1 + np.exp(-v))
    logit = lambda v: np.log(v / (1 - v))

    def unpack(x):      # pi_m, theta_p, w_hit, w_opp in (0,1); alpha > 1; kappa, sigma_f > 0
        return (sig(x[0]), 1 + np.exp(x[1]), np.exp(x[2]), np.exp(x[3]), sig(x[4]), sig(x[5]), sig(x[6]))

    def set_(x):
        m.pi_m, m.alpha, m.kappa, m.sigma_f, m.theta_p, m.w_hit, m.w_opp = unpack(x)

    def loss(x):
        set_(x)
        return float(dyn_loss(dyn_moments(m), tg))

    start = np.array([logit(d0.get("pi_m", 0.9)), np.log(d0.get("alpha", 1.5) - 1), np.log(d0.get("kappa", 0.015)),
                      np.log(d0.get("sigma_f", 2.5)), logit(d0.get("theta_p", 0.3)), logit(d0.get("w_hit", 0.3)),
                      logit(d0.get("w_opp", 0.3))])
    simplex = start + np.vstack([np.zeros(7), np.diag([0.6, 0.5, 0.7, 0.5, 0.8, 0.6, 0.6])])
    best = minimize(loss, start, method="Nelder-Mead", options={"maxfev": iters, "xatol": 0.02, "fatol": 1e-3,
                                                                 "initial_simplex": simplex})
    set_(best.x)
    pi_m, alpha, kappa, sigma_f, theta_p, w_hit, w_opp = unpack(best.x)
    fresh = dyn_moments(m, seeds=tuple(range(301, 311)))
    p["dyn"] = {"pi_m": pi_m, "alpha": alpha, "kappa": kappa, "sigma_f": sigma_f, "theta_p": theta_p,
                "w_hit": w_hit, "w_opp": w_opp, "loss": best.fun, "fit_fresh_seeds": fresh, "target": tg}
    print(f"dyn: pi_m {pi_m:.3f}  alpha {alpha:.2f}  kappa {kappa:.4f}/tick  sigma_f {sigma_f:.2f} t/sqrt(s)  theta_p {theta_p:.2f}"
          f"  w_hit {w_hit:.2f}  w_opp {w_opp:.2f}  (criterion {best.fun:.1f} on {len(DYN)} moments)")
    for k in DYN:
        tstat = (fresh[k] - tg[k]) / tg["se"][k]
        print(f"    {k:8s} data {tg[k]:7.3f} +- {tg['se'][k]:.3f}   sim (fresh seeds) {fresh[k]:7.3f}   t = {tstat:+.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("calib")
    ap.add_argument("--stage", choices=["static", "dyn", "all", "dyn+static"], default="all")
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()
    p = json.load(open(args.calib))
    stages = {"all": ["static", "dyn", "static"], "dyn+static": ["dyn", "static"]}.get(args.stage, [args.stage])
    for st in stages:
        print(f"== stage {st}")
        (stage_static if st == "static" else stage_dyn)(p, args.iters)
        json.dump(p, open(args.calib, "w"))


if __name__ == "__main__":
    main()
