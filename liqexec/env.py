"""Execution environment (deck slides 9, 11, 12): sell X shares within T seconds in the calibrated LOB.

Gymnasium interface; one episode is one parent order.

Liquidity schedule (`schedule`)
    "L" / "M" / "H"   fixed regime (Experiments A-C)
    "HML"             scripted deterioration H -> M -> L at T/3 and 2T/3 (Experiment D)
    "shock"           H for the first half, then L with 50% of displayed depth withdrawn at once
    "markov"          regime switches at each decision with the data's transition matrix
                      (per 5-minute window), scaled to the decision interval
    "mixed"           each episode draws one of L, M, H, HML, shock, markov (training distribution)

Observation (13 features, all observable to a real trader)
    0 remaining inventory q/X        1 elapsed time t/T             2 spread (ticks / 20)
    3 bid depth level 1 / slice      4 bid depth levels 1-5 / slice 5 ask depth levels 1-5 / slice
    6 depth imbalance (5 levels)     7 realised volatility of 1-s mids over the last step (ticks / 5)
    8 order-flow imbalance over the last step / (traded volume + slice)
    9 market volume over the last step / slice        10 mid move since arrival (bps / 10)
    11 share of the last limit order that filled       12 shortfall so far on executed shares (bps / 10)
    (slice = X / N; depth ratios are divided by 10 and clipped.) With oracle=True three more features
    give the true regime one-hot: an ablation of what observing the latent liquidity would be worth
    (deck slide 3; Macri and Lillo, 2024, argue it is latent).

Actions (18): child size m * X/N with m in {0, .5, 1, 1.5, 2, 3}, times style in
    {market order, limit at the best ask (join the queue), limit one tick inside the spread (improve)}.
A limit order lives for one interval and is cancelled at the next decision; its unfilled part stays in
inventory. At T, the remainder is sold with a market order (forced completion).

Reward (corrects the sign of slide 9's R_t = -(v_t P_t + lambda sigma^2 q_t^2); for a seller, v_t P_t is
revenue, so that reward would punish selling)
    r_k = 1e4 / (X S_0) * sum_fills q (p - S_0)  -  fees in bps  -  lam * (q_{k+1} / X)^2
so sum_k r_k = -IS (bps, net of fees) - risk penalty. lam is in bps per interval at full inventory.

Info at the end of an episode: implementation shortfall and its decomposition (Perold, 1988):
    delay   = sum q (S_0 - m_dec) : price moved before our order arrived (includes our own permanent impact)
    spread  = sum q (m_dec - b_dec) for market fills : half-spread paid at the touch
    impact  = sum q (b_dec - p) for market fills : walking the book beyond the touch
    passive = sum q (m_dec - p) for limit fills  : negative = spread captured
    fees    = taker fees - maker rebates
and the voluntary completion rate (share sold before the forced sweep).
"""
import json

import gymnasium as gym
import numpy as np
from scipy.linalg import fractional_matrix_power

from .lobsim import Market

SIZES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
STYLES = ("MO", "JOIN", "IMPROVE")
N_ACTIONS = len(SIZES) * len(STYLES)
OBS_DIM = 13
REG = {"L": 0, "M": 1, "H": 2}


class ExecutionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, calib="out/calib/regimes.json", schedule="M", X=6000, T=600.0, N=20, latency=0.05,
                 taker_fee=0.0030, maker_rebate=0.0020, lam=0.5, oracle=False, seed=0, market=None):
        super().__init__()
        self.mkt = market or Market(calib)
        self.schedule, self.X, self.T, self.N = schedule, X, T, N
        self.tau = T / N
        self.latency, self.lam, self.oracle = latency, lam, oracle
        tick = self.mkt.tick
        self.fee_t, self.reb_m = taker_fee / tick, maker_rebate / tick            # in ticks per share
        self.slice = X / N
        self.observation_space = gym.spaces.Box(-10, 10, (OBS_DIM + 3 * oracle,), np.float32)
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        P = np.array(self.mkt.calib["transition_per_window"])
        self.P_step = np.real(fractional_matrix_power(P, self.tau / self.mkt.calib["window_s"]))
        self.P_step = np.clip(self.P_step, 0, None)
        self.P_step /= self.P_step.sum(1, keepdims=True)
        self.rng = np.random.default_rng(seed)
        self.episode = 0

    # ------------------------------------------------------------------ schedule
    def _make_schedule(self, sched):
        """List of (time, regime, shock_frac) events; the first is at t = 0."""
        if sched == "mixed":
            sched = self.rng.choice(["L", "M", "H", "HML", "shock", "markov"])
        self.sched_name = sched
        if sched in REG:
            return [(0.0, REG[sched], 0.0)]
        if sched == "HML":
            return [(0.0, 2, 0.0), (self.T / 3, 1, 0.0), (2 * self.T / 3, 0, 0.0)]
        if sched == "shock":
            return [(0.0, 2, 0.0), (self.T / 2, 0, 0.5)]
        if sched == "markov":
            r = int(self.rng.integers(3))
            ev = [(0.0, r, 0.0)]
            for k in range(1, self.N):
                r2 = int(self.rng.choice(3, p=self.P_step[r]))
                if r2 != r:
                    ev.append((k * self.tau, r2, 0.0))
                r = r2
            return ev
        raise ValueError(sched)

    # ------------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            ep_seed = seed
        else:
            ep_seed = int(self.rng.integers(2 ** 31 - 1))
        self.ep_seed = ep_seed
        self.events = self._make_schedule((options or {}).get("schedule", self.schedule))
        m = self.mkt
        m.reset(self.events[0][1], seed=ep_seed)
        self.t0 = m.t
        self.ev_i = 1
        self.S0 = m.mid()
        self.q, self.k = self.X, 0
        self.cash_bps = 0.0
        self.comp = dict(delay=0.0, spread=0.0, impact=0.0, passive=0.0, fees=0.0)
        self.limit_qty, self.limit_fill = 0, 0.0
        self.voluntary = 0
        self.actions = []
        self.flow = (0, 0, 0.0, np.array([m.mid()]))
        m.clear_counters()
        self.episode += 1
        return self._obs(), {}

    def decode(self, a):
        return SIZES[a // len(STYLES)], STYLES[a % len(STYLES)]

    def _advance(self, t_end):
        """Run the market to t_end, applying scheduled regime switches on the way."""
        m = self.mkt
        while self.ev_i < len(self.events) and self.t0 + self.events[self.ev_i][0] <= t_end:
            te, r, shock = self.events[self.ev_i]
            m.run_until(self.t0 + te)
            m.set_regime(r, shock)
            self.ev_i += 1
        m.run_until(t_end)

    def _book_fills(self, fills, m_dec, b_dec):
        """Account fills (qty, price ticks, maker flag) against the decision-time mid and bid."""
        bps = 1e4 / (self.X * self.S0)
        r = 0.0
        sold = 0
        for qty, px, maker in fills:
            sold += qty
            r += qty * (px - self.S0) * bps
            self.comp["delay"] += qty * (self.S0 - m_dec) * bps
            if maker:
                self.comp["passive"] += qty * (m_dec - px) * bps
                fee = -self.reb_m * qty * bps
            else:
                self.comp["spread"] += qty * (m_dec - b_dec) * bps
                self.comp["impact"] += qty * (b_dec - px) * bps
                fee = self.fee_t * qty * bps
            self.comp["fees"] += fee
            r -= fee
        return r, sold

    def step(self, action, override=None):
        """override = (shares, style) lets scripted baselines bypass the discrete action set."""
        m = self.mkt
        t_dec = self.t0 + self.k * self.tau
        m_dec, b_dec = m.mid(), m.best_bid()
        if override is not None:
            qty, style = int(round(min(self.q, max(0.0, override[0])))), override[1]
        else:
            mult, style = self.decode(int(action))
            qty = int(round(min(self.q, mult * self.slice)))
        self.actions.append((self.k, qty, style))
        self._advance(t_dec + self.latency)                       # order travels to the exchange
        if qty > 0:
            if style == "MO":
                m.agent_market(qty)
            else:
                m.agent_limit(qty, improve=(style == "IMPROVE"))
        self._advance(t_dec + self.tau)
        left = m.cancel_agent()
        fills = m.take_fills()
        r, sold = self._book_fills(fills, m_dec, b_dec)
        self.q -= sold
        self.voluntary += sold
        self.limit_fill = (1 - left / qty) if (qty > 0 and style != "MO") else 0.0
        self.k += 1
        done = self.k == self.N
        if done and self.q > 0:                                   # forced completion at T
            m_T, b_T = m.mid(), m.best_bid()
            m.agent_market(self.q)
            r2, sold2 = self._book_fills(m.take_fills(), m_T, b_T)
            r += r2
            self.q -= sold2
        self.cash_bps += r
        r -= self.lam * (self.q / self.X) ** 2
        self.flow = m.take_flow()
        info = {}
        if done:
            info = {"is_bps": -self.cash_bps, "completion": self.voluntary / self.X, "schedule": self.sched_name,
                    **{f"c_{k}": v for k, v in self.comp.items()}}
        return self._obs(), float(r), done, False, info

    # ------------------------------------------------------------------ observation
    def _obs(self):
        m = self.mkt
        buy, sell, ofi, mids = self.flow
        sl = self.slice
        bid5, ask5 = m.depth(0, 5), m.depth(1, 5)
        vol = np.std(np.diff(mids)) if len(mids) > 2 else 0.0
        mid = m.mid()
        traded_px = 0.0
        if self.X - self.q > 0:
            traded_px = -(self.cash_bps) / ((self.X - self.q) / self.X)        # bps on executed shares
        o = [self.q / self.X, self.k / self.N, (m.best_ask() - m.best_bid()) / 20.0,
             min(m.depth(0, 1) / sl, 10) / 10, min(bid5 / sl, 10) / 10, min(ask5 / sl, 10) / 10,
             (bid5 - ask5) / max(bid5 + ask5, 1), vol / 5.0, ofi / (buy + sell + sl), (buy + sell) / sl / 10,
             (mid - self.S0) / self.S0 * 1e4 / 10, self.limit_fill, traded_px / 10]
        if self.oracle:
            o += list(np.eye(3)[m.r])
        return np.clip(np.array(o, dtype=np.float32), -10, 10)

    def regime(self):
        return self.mkt.r

    def features(self):
        """Raw liquidity features for the adaptive Almgren-Chriss baseline's regime classifier."""
        m = self.mkt
        return m.best_ask() - m.best_bid(), (m.depth(0, 5) + m.depth(1, 5)) / 2
