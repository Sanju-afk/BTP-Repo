"""Data-calibration pipeline (deck slide 10): LOBSTER message + orderbook files -> regime parameters.

    python3 -m liqexec.calibrate data/lobster/AAPL_..._message_10.csv [more message files] --out out/calib

Steps
  1. Split the day into W-second windows and measure liquidity per window: time-weighted spread,
     depth on the top 5 levels, visible market-order volume, realised volatility, order-flow imbalance.
  2. Liquidity score = z(log depth_5) - z(log spread); terciles of the score -> Low / Medium / High.
     Consecutive-window transitions give the empirical regime transition matrix.
  3. For each regime, estimate a level-based queue-reactive model (Huang, Lehalle and Rosenbaum, 2015):
     the intensity of limit orders, cancellations and market orders at each of the K occupied levels,
     as a function of that queue's size, is (number of events from state b) / (time spent in state b).
     That is the maximum-likelihood estimate for a Markov jump process. Bid and ask sides are pooled
     (symmetry assumption). Inserts inside the spread are a function of the spread; limit orders that
     open a new price between occupied levels are a constant rate with an empirical slot distribution.
  4. Empirical size distributions (limit, cancel, market order), gaps between levels, deep-level sizes,
     book snapshots for episode initialisation.
  5. Market-order impact in the data: mid change after each aggregated market order, by size.

AAPL is a small-tick stock (median spread 15 ticks, occupied levels several ticks apart), so queues are
indexed by occupied level, not by tick distance from a reference price as in the original QRM.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

K = 10                                                     # levels in the simulator
Q_EDGES = np.array([1, 100, 200, 300, 500, 800, 1300])     # queue-size bins (shares); bin b = [e_b, e_{b+1})
S_EDGES = np.array([1, 2, 3, 5, 8, 12, 16, 21, 31])        # spread bins (ticks)
TICK = 100                                                 # LOBSTER price units per tick ($0.01)
REGIMES = ["L", "M", "H"]
N_SAMPLES = 4000                                           # stored size draws per distribution
N_SNAP = 400                                               # stored book snapshots per regime
MIN_EXPOSURE = 20.0                                        # seconds; below this a bin borrows the level's pooled rate


def qbin(q):
    return np.clip(np.searchsorted(Q_EDGES, q, side="right") - 1, 0, len(Q_EDGES) - 1)


def sbin(s):
    return np.clip(np.searchsorted(S_EDGES, s, side="right") - 1, 0, len(S_EDGES) - 1)


def load(msg_path):
    ob_path = msg_path.replace("_message_", "_orderbook_")
    m = pd.read_csv(msg_path, header=None, usecols=range(6), names=["t", "type", "id", "size", "price", "dir"])
    ob = pd.read_csv(ob_path, header=None).to_numpy(np.int64)
    nlev = ob.shape[1] // 4
    if nlev < K:
        raise SystemExit(f"{ob_path}: need at least {K} levels, found {nlev}")
    ap, asz = ob[:, 0:4 * K:4] // TICK, ob[:, 1:4 * K:4]
    bp, bsz = ob[:, 2:4 * K:4] // TICK, ob[:, 3:4 * K:4]
    # LOBSTER marks missing levels with +-9999999999 and size 0
    ap = np.where(asz > 0, ap, 10 ** 8); bp = np.where(bsz > 0, bp, -10 ** 8)
    return m, ap, asz, bp, bsz


def window_features(t, ap, asz, bp, bsz, m, W):
    """Per-window liquidity measures. Returns a DataFrame indexed by window number."""
    dt = np.diff(t, append=t[-1])
    w = ((t - t[0]) // W).astype(int)
    spread = (ap[:, 0] - bp[:, 0]).astype(float)
    depth5 = (asz[:, :5].sum(1) + bsz[:, :5].sum(1)) / 2.0
    mid = (ap[:, 0] + bp[:, 0]) / 2.0
    q1 = (asz[:, 0] + bsz[:, 0]) / 2.0
    df = pd.DataFrame({"w": w, "dt": dt, "sp": spread * dt, "d5": depth5 * dt, "q1": q1 * dt})
    g = df.groupby("w").sum()
    out = pd.DataFrame({"spread": g.sp / g.dt, "depth5": g.d5 / g.dt, "q1": g.q1 / g.dt, "seconds": g.dt})
    vis = m.type.to_numpy() == 4
    out["mo_volume"] = pd.Series(m["size"].to_numpy()[vis]).groupby(w[vis]).sum()
    out["hidden_volume"] = pd.Series(m["size"].to_numpy()[m.type.to_numpy() == 5]).groupby(w[m.type.to_numpy() == 5]).sum()
    # realised volatility from mids on a 1-second grid, in ticks per sqrt(second)
    grid = np.arange(t[0], t[-1], 1.0)
    idx = np.searchsorted(t, grid, side="right") - 1
    r = np.diff(mid[idx])
    gw = ((grid[1:] - t[0]) // W).astype(int)
    out["vol"] = pd.Series(r).groupby(gw).std()
    out["n_events"] = pd.Series(np.ones(len(t))).groupby(w).sum()
    return out.fillna(0.0)


def label_regimes(feat):
    z = lambda x: (x - x.mean()) / x.std()
    score = z(np.log(feat.depth5)) - z(np.log(feat.spread))
    q1, q2 = np.quantile(score, [1 / 3, 2 / 3])
    lab = np.where(score <= q1, 0, np.where(score <= q2, 1, 2))
    return score, lab


def transition_matrix(lab, day):
    T = np.zeros((3, 3))
    for a, b, d1, d2 in zip(lab[:-1], lab[1:], day[:-1], day[1:]):
        if d1 == d2:
            T[a, b] += 1
    return T / np.maximum(T.sum(1, keepdims=True), 1)


class Accum:
    """Event counts and state exposure for one regime."""

    def __init__(self):
        B, SB = len(Q_EDGES), len(S_EDGES)
        self.exp_q = np.zeros((K, B))          # side-seconds spent with level k in bin b (sides pooled)
        self.n_lo = np.zeros((K, B)); self.n_cx = np.zeros((K, B)); self.n_mo = np.zeros(B)
        self.exp_s = np.zeros(SB); self.n_ins = np.zeros(SB)
        self.n_new = np.zeros(K); self.seconds = 0.0
        self.lo, self.cx, self.mo, self.gap, self.deep, self.ins_frac = [], [], [], [], [], []
        self.snaps = []


def accumulate(acc_by_reg, reg_of_row, t, m, ap, asz, bp, bsz, rng):
    n = len(t)
    dt = np.diff(t, append=t[-1])
    typ, size, price, d = (m[c].to_numpy() for c in ("type", "size", "price", "dir"))
    price = price // TICK
    spread = ap[:, 0] - bp[:, 0]
    for r, acc in enumerate(acc_by_reg):
        rows = np.nonzero(reg_of_row == r)[0]
        if len(rows) == 0:
            continue
        acc.seconds += dt[rows].sum()
        for side_sz in (asz, bsz):                       # exposure: state during [t_i, t_{i+1}) is row i
            for k in range(K):
                q = side_sz[rows, k]
                ok = q > 0
                np.add.at(acc.exp_q[k], qbin(q[ok]), dt[rows][ok])
        np.add.at(acc.exp_s, sbin(spread[rows]), dt[rows])

        ev = rows[rows > 0]                               # events: pre-state is row i-1
        pre = ev - 1
        is_bid = d[ev] == 1
        P = np.where(is_bid[:, None], bp[pre], ap[pre])
        Q = np.where(is_bid[:, None], bsz[pre], asz[pre])
        p = price[ev]
        eq = P == p[:, None]
        at = eq.any(1)
        lev = np.argmax(eq, 1)
        better_than = np.where(is_bid[:, None], P > p[:, None], P < p[:, None])
        n_better = better_than.sum(1)
        tp = typ[ev]
        qpre = Q[np.arange(len(ev)), lev]

        sel = (tp == 1) & at
        np.add.at(acc.n_lo, (lev[sel], qbin(qpre[sel])), 1)
        sel = np.isin(tp, (2, 3)) & at
        np.add.at(acc.n_cx, (lev[sel], qbin(qpre[sel])), 1)
        acc.cx += list(size[ev][sel])
        sel = (tp == 1) & ~at & (n_better == 0)            # insert inside the spread
        np.add.at(acc.n_ins, sbin(spread[pre][sel]), 1)
        sp_pre = spread[pre][sel]
        best = P[sel, 0]
        imp = np.abs(p[sel] - best)
        acc.ins_frac += list(np.clip(imp / np.maximum(sp_pre - 1, 1), 0, 1))
        sel = (tp == 1) & ~at & (n_better > 0) & (n_better < K)   # new price between occupied levels
        np.add.at(acc.n_new, n_better[sel], 1)
        acc.lo += list(size[ev][tp == 1])

        # market orders: consecutive visible executions with the same timestamp and direction
        ex = ev[tp == 4]
        if len(ex):
            new_mo = np.r_[True, (np.diff(ex) != 1) | (t[ex][1:] != t[ex][:-1]) | (d[ex][1:] != d[ex][:-1])]
            starts = ex[new_mo]
            sizes = np.add.reduceat(size[ex], np.nonzero(new_mo)[0])
            acc.mo += list(sizes)
            hit_bid = d[starts] == 1
            q1 = np.where(hit_bid, bsz[starts - 1, 0], asz[starts - 1, 0])
            np.add.at(acc.n_mo, qbin(q1), 1)

        # level gaps (ticks) and deep-level sizes for appending levels behind level K
        sub = rng.choice(rows, min(len(rows), 20000), replace=False)
        for Pq, Sq, sgn in ((ap, asz, 1), (bp, bsz, -1)):
            g = sgn * np.diff(Pq[sub], axis=1)
            ok = (Sq[sub, 1:] > 0) & (Sq[sub, :-1] > 0)
            acc.gap += list(g[ok][g[ok] > 0])
            acc.deep += list(Sq[sub, K - 1][Sq[sub, K - 1] > 0])
        for i in rng.choice(rows, min(len(rows), N_SNAP), replace=False):
            if (asz[i] > 0).all() and (bsz[i] > 0).all() and ap[i, 0] > bp[i, 0]:
                ref = bp[i, 0]
                acc.snaps.append(np.stack([bp[i] - ref, bsz[i], ap[i] - ref, asz[i]]))


def rates(acc):
    """Intensities per side (events / side-second). Sparse bins borrow the level's pooled rate."""
    def fill(n, e):
        lam = np.zeros_like(e)
        for k in range(e.shape[0]):
            pooled = n[k].sum() / max(e[k].sum(), 1e-9)
            lam[k] = np.where(e[k] >= MIN_EXPOSURE, n[k] / np.maximum(e[k], 1e-9), pooled)
        return lam
    lam_lo = fill(acc.n_lo, acc.exp_q)
    lam_cx = fill(acc.n_cx, acc.exp_q)
    lam_mo = fill(acc.n_mo[None], acc.exp_q[:1])[0]
    # inserts: per side, as a function of the spread; spread 1 tick has no room
    lam_ins = np.where(acc.exp_s >= MIN_EXPOSURE, acc.n_ins / np.maximum(2 * acc.exp_s, 1e-9),
                       acc.n_ins.sum() / max(2 * acc.exp_s.sum(), 1e-9))
    lam_ins[0] = 0.0
    lam_new = acc.n_new.sum() / (2 * acc.seconds)
    p_new = acc.n_new / max(acc.n_new.sum(), 1)
    return lam_lo, lam_cx, lam_mo, lam_ins, lam_new, p_new


def sample(x, rng, n=N_SAMPLES, cap=None):
    x = np.asarray(x, dtype=np.int64)
    if cap is not None:
        x = x[x <= cap]
    return rng.choice(x, n).tolist()


def mo_impact(t, m, ap, bp):
    """Mid change (ticks) right after, 10 s and 60 s after each aggregated market order, signed by direction."""
    typ, d, size = m.type.to_numpy(), m.dir.to_numpy(), m["size"].to_numpy()
    mid = (ap[:, 0] + bp[:, 0]) / 2.0
    ex = np.nonzero(typ == 4)[0]
    ex = ex[ex > 0]
    new_mo = np.r_[True, (np.diff(ex) != 1) | (t[ex][1:] != t[ex][:-1]) | (d[ex][1:] != d[ex][:-1])]
    first = ex[new_mo]
    last = np.r_[ex[np.nonzero(new_mo)[0][1:] - 1], ex[-1]]
    vol = np.add.reduceat(size[ex], np.nonzero(new_mo)[0])
    sign = np.where(d[first] == 1, -1.0, 1.0)            # hitting the bid = seller-initiated
    m0 = mid[first - 1]
    out = {"size": vol.tolist(), "imm": (sign * (mid[last] - m0)).tolist()}
    for h in (10, 60):
        j = np.minimum(np.searchsorted(t, t[last] + h), len(t) - 1)
        out[f"h{h}"] = (sign * (mid[j] - m0)).tolist()
    return out


LAGS = (10, 120, 300)


def dynamics(t, m, ap, bp):
    """Dynamic moments for the resilience stage: market-order response function R(l) (Bouchaud et al., 2004),
    lag-1 autocorrelation of market-order signs, and variance ratios of 1-second mids."""
    typ, d = m.type.to_numpy(), m.dir.to_numpy()
    mid = (ap[:, 0] + bp[:, 0]) / 2.0
    ex = np.nonzero(typ == 4)[0]
    ex = ex[ex > 0]
    new = np.r_[True, (np.diff(ex) != 1) | (t[ex][1:] != t[ex][:-1]) | (d[ex][1:] != d[ex][:-1])]
    first = ex[new]
    sign = np.where(d[first] == 1, -1.0, 1.0)
    m0 = mid[first - 1]
    out = {}
    for h in LAGS:
        j = np.minimum(np.searchsorted(t, t[first] + h), len(t) - 1)
        out[f"R{h}"] = float(np.mean(sign * (mid[j] - m0)))
    for l in (1, 5, 10):
        out[f"acf{l}"] = float(np.corrcoef(sign[:-l], sign[l:])[0, 1])
    g = np.arange(t[0], t[-1], 1.0)
    mm = mid[np.searchsorted(t, g, side="right") - 1]
    v1 = np.var(np.diff(mm))
    for h in (60, 300):
        out[f"VR{h}"] = float(np.var(mm[h:] - mm[:-h]) / (h * v1))
    return out


def dynamics_se(t, m, ap, bp, W=300.0, n_boot=500, seed=0):
    """Block-bootstrap standard errors of the dynamic moments, resampling W-second windows."""
    typ, d = m.type.to_numpy(), m.dir.to_numpy()
    mid = (ap[:, 0] + bp[:, 0]) / 2.0
    ex = np.nonzero(typ == 4)[0]
    ex = ex[ex > 0]
    new = np.r_[True, (np.diff(ex) != 1) | (t[ex][1:] != t[ex][:-1]) | (d[ex][1:] != d[ex][:-1])]
    first = ex[new]
    sign = np.where(d[first] == 1, -1.0, 1.0)
    m0 = mid[first - 1]
    wmo = ((t[first] - t[0]) // W).astype(int)
    nw = wmo.max() + 1
    per = {}
    for h in LAGS:
        j = np.minimum(np.searchsorted(t, t[first] + h), len(t) - 1)
        per[f"R{h}"] = np.bincount(wmo, sign * (mid[j] - m0), nw)
    per["n"] = np.bincount(wmo, minlength=nw).astype(float)
    for l in (1, 5, 10):
        same_w = wmo[:-l] == wmo[l:]
        per[f"c{l}"] = np.bincount(wmo[:-l][same_w], (sign[:-l] * sign[l:])[same_w], nw)
        per[f"n{l}"] = np.bincount(wmo[:-l][same_w], minlength=nw).astype(float)
    g = np.arange(t[0], t[-1], 1.0)
    mm = mid[np.searchsorted(t, g, side="right") - 1]
    wg = ((g - t[0]) // W).astype(int)
    for h in (1, 60, 300):
        r = mm[h:] - mm[:-h]
        per[f"v{h}"] = np.bincount(wg[:-h], r ** 2, nw)
        per[f"nv{h}"] = np.bincount(wg[:-h], minlength=nw).astype(float)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, nw, nw)
        S = {k: v[idx].sum() for k, v in per.items()}
        R10 = S["R10"] / S["n"]
        v1 = S["v1"] / S["nv1"]
        boots.append({"R10": R10, "rev120": S["R120"] / S["n"] / R10, "rev300": S["R300"] / S["n"] / R10,
                      "acf1": S["c1"] / S["n1"], "acf5": S["c5"] / S["n5"], "acf10": S["c10"] / S["n10"],
                      "VR60": S["v60"] / S["nv60"] / (60 * v1), "VR300": S["v300"] / S["nv300"] / (300 * v1)})
    return {k: float(np.std([b[k] for b in boots])) for k in boots[0]}


def fit_hawkes(t, m):
    """Univariate Hawkes process with exponential kernel, by maximum likelihood, on market-order times:
    lambda(t) = mu + alpha * sum exp(-beta (t - t_i)). Returns mu, alpha, beta and branching ratio n = alpha/beta."""
    from scipy.optimize import minimize
    typ, d = m.type.to_numpy(), m.dir.to_numpy()
    ex = np.nonzero(typ == 4)[0]
    new = np.r_[True, (np.diff(ex) != 1) | (t[ex][1:] != t[ex][:-1]) | (d[ex][1:] != d[ex][:-1])]
    T = t[ex[new]] - t[ex[new]][0]
    gaps = np.diff(T)

    def nll(x):
        mu, a, b = np.exp(x)
        A = np.zeros(len(T))
        decay = np.exp(-b * gaps)
        for i in range(1, len(T)):
            A[i] = decay[i - 1] * (1 + A[i - 1])
        comp = mu * T[-1] + (a / b) * np.sum(1 - np.exp(-b * (T[-1] - T)))
        return -(np.sum(np.log(mu + a * A)) - comp)

    r = minimize(nll, np.log([0.3, 50.0, 100.0]), method="Nelder-Mead", options={"maxfev": 3000, "xatol": 1e-4})
    mu, a, b = np.exp(r.x)
    return {"mu": float(mu), "alpha": float(a), "beta": float(b), "n": float(a / b), "n_events": int(len(T))}


def fit_power(size, dmid, nbins=10):
    """Fit E[dmid | size] = a * size^delta on log-spaced size bins (bins with a positive mean)."""
    size, dmid = np.asarray(size, float), np.asarray(dmid, float)
    edges = np.unique(np.quantile(size, np.linspace(0, 1, nbins + 1)))
    b = np.clip(np.searchsorted(edges, size, side="right") - 1, 0, len(edges) - 2)
    xs, ys, ns = [], [], []
    for i in range(len(edges) - 1):
        s = b == i
        if s.sum() >= 30 and dmid[s].mean() > 0:
            xs.append(size[s].mean()); ys.append(dmid[s].mean()); ns.append(int(s.sum()))
    if len(xs) < 3:
        return None
    delta, loga = np.polyfit(np.log(xs), np.log(ys), 1)
    return {"delta": float(delta), "a": float(np.exp(loga)), "x": xs, "y": ys, "n": ns}


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("messages", nargs="+", help="LOBSTER message files (orderbook files found by name)")
    ap_.add_argument("--window", type=float, default=300.0)
    ap_.add_argument("--skip-open", type=float, default=300.0, help="seconds dropped after the open (auction effects)")
    ap_.add_argument("--out", default="out/calib")
    ap_.add_argument("--seed", type=int, default=0)
    args = ap_.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    days = []
    for path in sorted(sum((glob.glob(p) for p in args.messages), [])):
        m, ap, asz, bp, bsz = load(path)
        t = m.t.to_numpy()
        keep = t >= t[0] + args.skip_open
        k0 = np.argmax(keep)
        m = m.iloc[k0:].reset_index(drop=True)
        ap, asz, bp, bsz, t = ap[k0:], asz[k0:], bp[k0:], bsz[k0:], t[k0:]
        feat = window_features(t, ap, asz, bp, bsz, m, args.window)
        feat["day"] = os.path.basename(path).split("_message")[0]
        days.append((path, m, t, ap, asz, bp, bsz, feat))
        print(f"{path}: {len(m)} messages, {len(feat)} windows")

    feat_all = pd.concat([d[-1] for d in days], keys=range(len(days)), names=["d", "w"])
    score, lab = label_regimes(feat_all)
    feat_all["score"], feat_all["regime"] = score, [REGIMES[i] for i in lab]
    Tm = transition_matrix(lab, feat_all.index.get_level_values("d"))

    dyn = [dynamics(d[2], d[1], d[3], d[5]) for d in days]
    data_dyn = {k: float(np.mean([x[k] for x in dyn])) for k in dyn[0]}
    se = [dynamics_se(d[2], d[1], d[3], d[5]) for d in days]
    data_dyn["se"] = {k: float(np.sqrt(np.mean([x[k] ** 2 for x in se])) / np.sqrt(len(se))) for k in se[0]}
    accs = [Accum() for _ in REGIMES]
    impact = {"size": [], "imm": [], "h10": [], "h60": []}
    for di, (path, m, t, ap, asz, bp, bsz, feat) in enumerate(days):
        w = ((t - t[0]) // args.window).astype(int)
        reg_of_w = dict(zip(feat_all.loc[di].index, lab[feat_all.index.get_level_values("d") == di]))
        reg_of_row = np.array([reg_of_w.get(x, -1) for x in w])
        accumulate(accs, reg_of_row, t, m, ap, asz, bp, bsz, rng)
        imp = mo_impact(t, m, ap, bp)
        for k in impact:
            impact[k] += imp[k]

    hk = [fit_hawkes(d[2], d[1]) for d in days]
    hawkes = {k: float(np.mean([x[k] for x in hk])) for k in ("mu", "alpha", "beta", "n")}
    params = {"data_dyn": data_dyn, "hawkes": hawkes, "K": K, "q_edges": Q_EDGES.tolist(), "s_edges": S_EDGES.tolist(), "tick_dollars": TICK / 1e4,
              "regimes": REGIMES, "window_s": args.window, "transition_per_window": Tm.tolist(), "per_regime": {}}
    for name, acc, (lam_lo, lam_cx, lam_mo, lam_ins, lam_new, p_new) in zip(REGIMES, accs, map(rates, accs)):
        f = feat_all[feat_all.regime == name]
        params["per_regime"][name] = {
            "lam_lo": lam_lo.tolist(), "lam_cx": lam_cx.tolist(), "lam_mo": lam_mo.tolist(),
            "lam_ins": lam_ins.tolist(), "lam_new": float(lam_new), "p_new": p_new.tolist(),
            "lo_size": sample(acc.lo, rng, cap=5000), "cx_size": sample(acc.cx, rng, cap=5000),
            "mo_size": sample(acc.mo, rng, cap=5000), "gap": sample(acc.gap, rng, cap=50),
            "deep_size": sample(acc.deep, rng, cap=5000), "ins_frac": rng.choice(acc.ins_frac, N_SAMPLES).tolist(),
            "snapshots": np.array(acc.snaps).tolist(),
            "data": {"windows": int(len(f)), "seconds": float(acc.seconds), "spread_ticks": float(f.spread.mean()),
                     "depth5": float(f.depth5.mean()), "q1": float(f.q1.mean()), "mo_volume_per_s": float(f.mo_volume.sum() / f.seconds.sum()),
                     "vol_ticks_per_sqrt_s": float(np.sqrt((f.vol ** 2 * f.seconds).sum() / f.seconds.sum())),
                     "events_per_s": float(f.n_events.sum() / f.seconds.sum()),
                     "mo_per_s": float(len(acc.mo) / acc.seconds)},
        }
    fit = {h: fit_power(impact["size"], impact[h]) for h in ("imm", "h10", "h60")}
    params["data_mo_impact"] = {h: (None if v is None else {k: v[k] for k in ("delta", "a", "x", "y", "n")}) for h, v in fit.items()}
    json.dump(params, open(os.path.join(args.out, "regimes.json"), "w"))
    feat_all.to_csv(os.path.join(args.out, "windows.csv"))

    print("\nregime  windows  spread(t)  depth5  MO vol/s  vol(t/sqrt s)  events/s")
    for name in REGIMES:
        dd = params["per_regime"][name]["data"]
        print(f"  {name}      {dd['windows']:4d}    {dd['spread_ticks']:6.1f}  {dd['depth5']:7.0f}  {dd['mo_volume_per_s']:7.1f}"
              f"   {dd['vol_ticks_per_sqrt_s']:8.2f}     {dd['events_per_s']:6.1f}")
    print("transition matrix per window (rows L, M, H):\n", np.round(Tm, 2))
    print("dynamics:", {k: round(v, 3) for k, v in data_dyn.items() if k != "se"})
    print("   s.e.:", {k: round(v, 3) for k, v in data_dyn["se"].items()})
    print(f"Hawkes (market orders): branching ratio {hawkes['n']:.2f}, kernel decay {1e3 / hawkes['beta']:.1f} ms")
    for h, v in fit.items():
        if v:
            print(f"single-MO impact {h}: delta = {v['delta']:.2f} (power-law fit on {len(v['x'])} size bins)")


if __name__ == "__main__":
    main()
