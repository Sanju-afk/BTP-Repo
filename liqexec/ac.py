"""Almgren-Chriss (2000): closed form, calibration inside the simulator, and the idealised AC world.

Calibration (deck slide 10: "impact parameters fitted, not assumed"). In each regime, an agent sells random
market-order sizes each interval. Regressions:
    per-share slippage vs the decision mid  = eps + eta * (n / tau)       (temporary impact, half-spread)
    mid change over the interval             = drift - gamma * n           (permanent impact)
    sigma = std of that regression's residual / sqrt(tau)
Also fitted: slippage = eps + a * (n / tau)^beta, to report how far the linear AC impact is from the
simulator's (deck slides 3-4: linear impact is a limitation of AC).

Idealised AC world (deck sub-question 4: do rankings change between an idealised AC simulator and the
data-calibrated one?). ACIdealEnv has the same interface, observation layout and actions as ExecutionEnv,
with arithmetic Brownian motion and linear impact using the calibrated parameters. Limit orders do not
exist there, so every style executes as a market order.
"""
import json
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

from .env import N_ACTIONS, OBS_DIM, REG, SIZES, STYLES, ExecutionEnv


@dataclass
class ACParams:
    X: float
    N: int
    tau: float
    sigma: float      # ticks per sqrt(s)
    gamma: float      # ticks per share
    eta: float        # ticks per (share/s)
    eps: float        # ticks (fixed cost per share)
    lam: float        # risk aversion, 1 / tick

    @property
    def eta_tilde(self):
        return max(self.eta - 0.5 * self.gamma * self.tau, 1e-12)

    @property
    def kappa(self):
        c = 1 + 0.5 * self.lam * self.sigma ** 2 * self.tau ** 2 / self.eta_tilde
        return math.acosh(c) / self.tau


def closed_form(p: ACParams):
    """Holdings x_0..x_N of the mean-variance optimal schedule."""
    T, k = p.N * p.tau, p.kappa
    if k * T < 1e-9:
        return np.array([p.X * (1 - j / p.N) for j in range(p.N + 1)])
    return np.array([p.X * math.sinh(k * (T - j * p.tau)) / math.sinh(k * T) for j in range(p.N + 1)])


def lam_from_penalty(lam_bps, S0, sigma, tau, X):
    """Map the RL penalty lam' (q/X)^2 bps per interval to AC's lambda sigma^2 tau q^2 (both in bps of X S0)."""
    return lam_bps * S0 / (1e4 * sigma ** 2 * tau * X)


def calibrate(calib="out/calib/regimes.json", episodes=300, seed=0, X=6000, T=600.0, N=20):
    out = {}
    for name, r in REG.items():
        env = ExecutionEnv(calib, schedule=name, X=X, T=T, N=N, lam=0.0, seed=seed)
        rng = np.random.default_rng(seed + r)
        ns, dmid, rate, slip = [], [], [], []
        for ep in range(episodes):
            env.reset(seed=50_000 + 1000 * r + ep)
            env.q = 10 ** 9                       # unlimited inventory: sizes are exogenous
            for k in range(N):
                n = int(rng.integers(0, 7)) * env.X // (2 * N)       # 0 .. 3 slices
                m0, t_dec = env.mkt.mid(), env.t0 + k * env.tau
                env._advance(t_dec + env.latency)
                if n:
                    env.mkt.agent_market(n)
                f = env.mkt.take_fills()
                env._advance(t_dec + env.tau)
                env.mkt.take_flow()
                ns.append(n); dmid.append(env.mkt.mid() - m0)
                if n:
                    vwap = (f[:, 0] * f[:, 1]).sum() / f[:, 0].sum()
                    rate.append(n / env.tau); slip.append(m0 - vwap)
        ns, dmid, rate, slip = map(np.array, (ns, dmid, rate, slip))
        A = np.c_[-ns, np.ones_like(ns)]
        (gamma, drift), *_ = np.linalg.lstsq(A, dmid, rcond=None)
        sigma = np.std(dmid - A @ [gamma, drift]) / math.sqrt(env.tau)
        B = np.c_[rate, np.ones_like(rate)]
        (eta, eps), *_ = np.linalg.lstsq(B, slip, rcond=None)
        try:
            (e2, a2, beta), _ = curve_fit(lambda v, e, a, b: e + a * v ** b, rate, slip, p0=[eps, eta, 1.0], maxfev=20000)
        except RuntimeError:
            e2 = a2 = beta = float("nan")
        out[name] = {"gamma": float(gamma), "eta": float(eta), "eps": float(eps), "sigma": float(sigma),
                     "tau": env.tau, "S0": float(env.S0), "power_fit": {"eps": float(e2), "a": float(a2), "beta": float(beta)}}
        print(f"{name}: gamma {gamma:.4f} t/share  eta {eta:.4f} t/(share/s)  eps {eps:.2f} t  sigma {sigma:.2f} t/sqrt(s)"
              f"  | slippage ~ (rate)^{beta:.2f}")
    return out


class ACIdealEnv:
    """Idealised Almgren-Chriss market with ExecutionEnv's interface (fixed or scripted regimes)."""

    def __init__(self, ac_params, schedule="M", X=6000, T=600.0, N=20, lam=0.5, taker_fee=0.0030, tick=0.01, seed=0,
                 calib="out/calib/regimes.json", oracle=False):
        self.pp = ac_params
        c = json.load(open(calib))
        self.vol = {n: c["per_regime"][n]["data"]["mo_volume_per_s"] for n in REG}     # for VWAP / POV baselines
        self.P_step = np.eye(3)
        self.schedule, self.X, self.T, self.N, self.lam = schedule, X, T, N, lam
        self.tau, self.slice = T / N, X / N
        self.fee_t = taker_fee / tick
        self.rng = np.random.default_rng(seed)
        self.observation_space_shape = (OBS_DIM,)

    def _regime_at(self, k):
        s = self.sched_name
        if s in REG:
            return s
        if s == "HML":
            return ["H", "M", "L"][min(2, int(3 * k / self.N))]
        if s == "shock":
            return "H" if k < self.N // 2 else "L"
        raise ValueError(s)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        s = (options or {}).get("schedule", self.schedule)
        if s == "mixed":
            s = self.rng.choice(["L", "M", "H", "HML", "shock"])
        self.sched_name = s
        self.S0 = self.pp[self._regime_at(0)]["S0"]
        self.S, self.q, self.k, self.cash_bps = self.S0, self.X, 0, 0.0
        self.actions = []
        v = self.vol[self._regime_at(0)] * self.tau / 2
        self.flow = (v, v)
        return self._obs(), {}

    def decode(self, a):
        return SIZES[a // len(STYLES)], STYLES[a % len(STYLES)]

    def _obs(self):
        o = np.zeros(OBS_DIM, np.float32)
        o[0], o[1] = self.q / self.X, self.k / self.N
        o[10] = (self.S - self.S0) / self.S0 * 1e4 / 10
        return o

    def features(self):
        return None

    def step(self, action, override=None):
        p = self.pp[self._regime_at(self.k)]
        if override is not None:
            n = min(self.q, max(0.0, override[0]))
        else:
            n = min(self.q, self.decode(int(action))[0] * self.slice)
        if self.k == self.N - 1:
            n = self.q
        n = float(round(n))
        self.actions.append((self.k, n, "MO"))
        bps = 1e4 / (self.X * self.S0)
        px = self.S - p["eps"] - p["eta"] * n / self.tau
        r = n * (px - self.S0) * bps - self.fee_t * n * bps
        self.S += p["sigma"] * math.sqrt(self.tau) * self.rng.standard_normal() - p["gamma"] * n
        self.q -= n
        self.k += 1
        self.cash_bps += r
        v = self.vol[self._regime_at(min(self.k, self.N - 1))] * self.tau / 2
        self.flow = (v, v)
        done = self.k == self.N
        r -= self.lam * (self.q / self.X) ** 2
        info = {"is_bps": -self.cash_bps, "completion": 1.0, "schedule": self.sched_name} if done else {}
        return self._obs(), float(r), done, False, info


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="out/calib/regimes.json")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--out", default="out/calib/ac_params.json")
    a = ap.parse_args()
    json.dump(calibrate(a.calib, a.episodes), open(a.out, "w"), indent=1)
