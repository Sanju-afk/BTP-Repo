"""Benchmark execution strategies (deck slide 12), plus two adaptive ones added because RL execution papers
are often compared only with TWAP (a weak baseline, Tonkin et al., 2025):

  TWAP           X/N per interval, market orders
  TWAP-LO        X/N per interval as a limit order at the best ask; the remainder is swept at T
  VWAP           shares proportional to forecast market volume per interval. The forecast is conditional:
                 the regime is classified at the start from observable spread and depth, and expected
                 volume in later intervals follows the data's regime transition matrix (dynamic volume
                 forecasting, Bialkowski et al., 2008). With no information it would equal TWAP.
  POV            sells a fixed share rho of the market volume seen in the last interval;
                 rho = X / forecast volume over T
  AC             Almgren-Chriss schedule with parameters calibrated in the regime classified at the start
  AC-adaptive    re-solves AC every interval for the remaining shares and time, with the parameters of the
                 regime currently classified from observables (a model-based adaptive benchmark)

All of them observe only what the RL agents observe (no regime label).
"""
import json

import numpy as np

from .ac import ACParams, closed_form, lam_from_penalty
from .env import REG

NAMES = ["L", "M", "H"]


class RegimeClassifier:
    """Nearest calibrated regime in standardised (log spread, log depth5), smoothed across intervals."""

    def __init__(self, calib, alpha=0.5):
        d = [calib["per_regime"][n]["data"] for n in NAMES]
        self.c = np.array([[np.log(x["spread_ticks"]), np.log(x["depth5"])] for x in d])
        self.scale = self.c.std(0) + 1e-9
        self.alpha = alpha
        self.z = None

    def reset(self):
        self.z = None

    def __call__(self, spread, depth5):
        x = np.array([np.log(max(spread, 1)), np.log(max(depth5, 1))])
        self.z = x if self.z is None else self.alpha * x + (1 - self.alpha) * self.z
        return int(np.argmin((((self.z - self.c) / self.scale) ** 2).sum(1)))


class Baselines:
    def __init__(self, calib_path="out/calib/regimes.json", ac_path="out/calib/ac_params.json"):
        self.calib = json.load(open(calib_path))
        self.ac = json.load(open(ac_path))
        self.vol = np.array([self.calib["per_regime"][n]["data"]["mo_volume_per_s"] for n in NAMES])
        self.clf = RegimeClassifier(self.calib)

    # ------------------------------------------------------------------ helpers
    def _start(self, env):
        """Called at k = 0: classify regime, compute forecasts."""
        self.clf.reset()
        f = env.features()
        self.r0 = self.clf(*f) if f is not None else REG[env._regime_at(0)]
        P = env.P_step if hasattr(env, "P_step") else np.eye(3)
        p = np.eye(3)[self.r0]
        ev = []
        for _ in range(env.N):
            ev.append(p @ self.vol * env.tau)
            p = p @ P
        self.exp_vol = np.array(ev)

    def _ac_schedule(self, env, regime_name, q, steps_left):
        c = self.ac[regime_name]
        lam = lam_from_penalty(env.lam, c["S0"], c["sigma"], c["tau"], env.X)
        p = ACParams(X=q, N=steps_left, tau=c["tau"], sigma=c["sigma"], gamma=max(c["gamma"], 0), eta=c["eta"],
                     eps=c["eps"], lam=lam)
        return closed_form(p)

    # ------------------------------------------------------------------ policies: (env) -> (shares, style)
    def twap(self, env):
        return env.X / env.N, "MO"

    def twap_lo(self, env):
        return env.X / env.N, "JOIN"

    def vwap(self, env):
        if env.k == 0:
            self._start(env)
        w = self.exp_vol[env.k:] / self.exp_vol[env.k:].sum()
        return env.q * w[0], "MO"

    def pov(self, env):
        if env.k == 0:
            self._start(env)
            self.rho = env.X / self.exp_vol.sum()
            return self.rho * self.exp_vol[0], "MO"
        mv = getattr(env, "flow", (0, 0))[0] + getattr(env, "flow", (0, 0))[1]
        return self.rho * mv, "MO"

    def ac_static(self, env):
        if env.k == 0:
            self._start(env)
            self.x_ac = self._ac_schedule(env, NAMES[self.r0], env.X, env.N)
        return self.x_ac[env.k] - self.x_ac[env.k + 1], "MO"

    def ac_adaptive(self, env):
        if env.k == 0:
            self._start(env)
            r = self.r0
        else:
            f = env.features()
            r = self.clf(*f) if f is not None else REG[env._regime_at(env.k)]
        x = self._ac_schedule(env, NAMES[r], env.q, env.N - env.k)
        return x[0] - x[1], "MO"

    def all(self):
        return {"TWAP": self.twap, "TWAP-LO": self.twap_lo, "VWAP": self.vwap, "POV": self.pov,
                "AC": self.ac_static, "AC-adaptive": self.ac_adaptive}
