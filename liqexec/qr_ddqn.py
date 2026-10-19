"""Quantile-regression Double DQN with risk-sensitive action selection (v2 extension).

QR-DQN (Dabney, Rowland, Bellemare and Munos, 2018, AAAI) learns K quantiles of the return Z(x, a) with the
quantile Huber loss. The target is Double-DQN style (van Hasselt et al., 2016): the online network picks the
next action with the same risk criterion used for acting, and the target network supplies its quantiles.
Everything else (MLP body 13 -> 128 -> 128, replay, warm-up, epsilon schedule, target sync, gradient clip,
step budget) is the Phase-1 DDQN in liqexec/agents.py, so the only change is the head and the criterion.

Risk criteria (`risk`), on the return Z = -(shortfall + risk penalty), higher is better:
    "mean"       risk-neutral: mean of the K quantiles (isolates the effect of learning a distribution)
    "cvar"       CVaR_alpha-greedy: mean of the lowest alpha*K quantiles (the worst-cost outcomes). This is a
                 per-step (myopic) risk-averse rule; it is NOT the policy that maximises the static CVaR of
                 the total return (Lim and Malik, 2022, Prop. 1), and conditioning on cost-so-far does not
                 change that because CVaR is translation-equivariant.
    "threshold"  static-CVaR rule of Lim and Malik (2022), after Baeuerle and Ott (2011): carry a threshold
                 c in the state, start at c_0 = VaR_alpha of the CVaR-best action's return at the initial
                 state, update c <- c - r after each reward (gamma = 1), and act by
                 argmax_a E[-(c - Z(x, c, a))^+]. The network sees c/10 as a 14th input feature.

Quantile midpoints tau_i = (2i - 1) / 2K. Loss: sum over predicted quantiles, mean over target quantiles
and batch (as in the QR-DQN paper). Quantiles are sorted before any risk statistic is taken.
"""
import copy
import random
import time
from collections import deque

import numpy as np
import torch
from torch import nn

from liqexec.agents import DDQN, mlp

torch.set_num_threads(1)


def quantile_huber_loss(pred, target, taus, kappa=1.0):
    """pred: (B, K) predicted quantiles; target: (B, K') target samples; taus: (K,) quantile levels.

    rho_tau^kappa(u) = |tau - 1{u < 0}| * Huber_kappa(u) / kappa with u = target_j - pred_i.
    Returns the scalar loss: sum over i, mean over j and batch.
    """
    u = target[:, None, :] - pred[:, :, None]                        # (B, K, K')
    absu = u.abs()
    huber = torch.where(absu <= kappa, 0.5 * u ** 2, kappa * (absu - 0.5 * kappa))
    w = (taus[None, :, None] - (u.detach() < 0).float()).abs()
    return (w * huber / kappa).mean(2).sum(1).mean()


def cvar_lower(z, alpha):
    """Mean of the lowest round(alpha*K) sorted quantiles along the last axis (lower tail of the return)."""
    k = z.shape[-1]
    m = max(1, int(round(alpha * k)))
    return torch.sort(z, dim=-1).values[..., :m].mean(-1)


def var_lower(z, alpha):
    """alpha-quantile of the return: the largest of the lowest alpha*K sorted quantiles."""
    k = z.shape[-1]
    m = max(1, int(round(alpha * k)))
    return torch.sort(z, dim=-1).values[..., m - 1]


def threshold_score(z, c):
    """E[-(c - Z)^+] for quantiles z (..., K) and threshold c broadcastable to z[..., 0]."""
    return -torch.clamp(c[..., None] - z, min=0.0).mean(-1)


def risk_score(z, risk, alpha, c=None):
    if risk == "mean":
        return z.mean(-1)
    if risk == "cvar":
        return cvar_lower(z, alpha)
    if risk == "threshold":
        return threshold_score(z, c)
    raise ValueError(risk)


class QRDDQN(DDQN):
    name = "QRDDQN"

    def __init__(self, obs_dim, n_actions, lr=1e-4, buffer=100_000, batch=128, sync=1000, eps_frac=0.5,
                 eps_end=0.05, gamma=1.0, train_freq=4, seed=0, n_quantiles=40, kappa=1.0, risk="cvar",
                 alpha=0.25, **_):
        torch.manual_seed(seed)
        self.risk, self.alpha, self.K, self.kappa = risk, alpha, n_quantiles, kappa
        self.extra = 1 if risk == "threshold" else 0
        self.in_dim = obs_dim + self.extra
        self.q = mlp(self.in_dim, n_actions * n_quantiles)
        self.tgt = copy.deepcopy(self.q)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buf = deque(maxlen=buffer)
        self.n_actions, self.batch, self.sync, self.gamma = n_actions, batch, sync, gamma
        self.train_freq = train_freq
        self.eps_frac, self.eps_end = eps_frac, eps_end
        self.rng = random.Random(seed)
        self.updates = 0
        self.taus = (torch.arange(n_quantiles, dtype=torch.float32) + 0.5) / n_quantiles
        self.c = 0.0                                        # threshold (risk == "threshold" only)

    # ---------------------------------------------------------------- distribution and acting
    def quantiles(self, x, net=None):
        net = net or self.q
        return net(x).view(x.shape[0], self.n_actions, self.K)

    def _aug(self, s, c):
        if not self.extra:
            return np.asarray(s, np.float32)
        return np.append(np.asarray(s, np.float32), np.float32(c / 10.0))

    @torch.no_grad()
    def begin_episode(self, s):
        """Reset the threshold at the start of an episode (no-op unless risk == 'threshold')."""
        if self.risk != "threshold":
            return
        self.c = 0.0
        for _ in range(2):                          # c is also a network input: one fixed-point refinement
            z = self.quantiles(torch.as_tensor(self._aug(s, self.c))[None])[0]
            a = int(cvar_lower(z, self.alpha).argmax().item())
            self.c = float(var_lower(z[a], self.alpha).item())

    def observe(self, r):
        if self.risk == "threshold":
            self.c = (self.c - r) / self.gamma

    @torch.no_grad()
    def act(self, s, greedy=True, eps=0.0):
        if not greedy and self.rng.random() < eps:
            return self.rng.randrange(self.n_actions)
        z = self.quantiles(torch.as_tensor(self._aug(s, self.c))[None])[0]
        c = torch.tensor(self.c, dtype=torch.float32).expand(self.n_actions)
        return int(risk_score(z, self.risk, self.alpha, c).argmax().item())

    # ---------------------------------------------------------------- learning
    def _update(self):
        b = self.rng.sample(self.buf, self.batch)
        s, a, r, s2, d, c2 = map(np.array, zip(*b))
        s, s2 = torch.as_tensor(s), torch.as_tensor(s2)
        a, r, d = torch.as_tensor(a), torch.as_tensor(r, dtype=torch.float32), torch.as_tensor(d, dtype=torch.float32)
        c2 = torch.as_tensor(c2, dtype=torch.float32)
        idx = torch.arange(len(a))
        z = self.quantiles(s)[idx, a]                                             # (B, K)
        with torch.no_grad():
            cc = c2[:, None].expand(-1, self.n_actions)
            a2 = risk_score(self.quantiles(s2), self.risk, self.alpha, cc).argmax(1)
            z2 = self.quantiles(s2, self.tgt)[idx, a2]
            y = r[:, None] + self.gamma * (1 - d)[:, None] * z2
        loss = quantile_huber_loss(z, y, self.taus, self.kappa)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        self.updates += 1
        if self.updates % self.sync == 0:
            self.tgt.load_state_dict(self.q.state_dict())

    def train(self, env, steps, log=None, seed=0):
        rng = np.random.default_rng(seed)
        t, ep, t0 = 0, 0, time.time()
        while t < steps:
            s, _ = env.reset(seed=int(rng.integers(2 ** 31 - 1)))
            self.begin_episode(s)
            done = False
            while not done:
                eps = max(self.eps_end, 1 - (1 - self.eps_end) * t / (self.eps_frac * steps))
                x = self._aug(s, self.c)
                a = self.act(s, greedy=False, eps=eps)
                s2, r, done, _, info = env.step(a)
                self.observe(r)
                x2 = self._aug(s2, self.c)
                self.buf.append((x, a, r, x2, float(done), self.c))
                s = s2
                t += 1
                if len(self.buf) >= 1000 and t % self.train_freq == 0:
                    self._update()
            ep += 1
            if log is not None:
                log.append((t, ep, info["is_bps"], time.time() - t0))


# Arms of the v2 study: name -> constructor overrides (hyperparameters come from tuning).
ARMS = {
    "QRDDQN-RN": dict(risk="mean"),
    "QRDDQN-CVaR25": dict(risk="cvar", alpha=0.25),
    "QRDDQN-CVaR10": dict(risk="cvar", alpha=0.10),
    "QRDDQN-THR25": dict(risk="threshold", alpha=0.25),
    # exploratory control (added after the main results, not pre-registered): risk-neutral with the CVaR25
    # arm's hyperparameters (train_freq 4), to separate the distribution from QRDDQN-RN's 4x update rate
    "QRDDQN-RN4": dict(risk="mean"),
}
