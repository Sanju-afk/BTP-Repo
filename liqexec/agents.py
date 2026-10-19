"""DDQN (value-based) and PPO (policy-gradient) under one protocol (deck slides 9 and 12).

Fairness rules (Tonkin et al., 2025; Henderson et al., 2018):
  * identical observation, action set (18 discrete actions) and reward;
  * identical network bodies: MLP obs -> 128 -> 128 (ReLU). DDQN has one Q head; PPO has an actor and a
    critic, each with that body;
  * identical budget of environment steps per run, and the same small tuning budget per algorithm
    (experiments/tune.py);
  * both evaluated deterministically (DDQN: argmax Q; PPO: argmax pi), on the same episodes (common seeds).
Both are undiscounted (gamma = 1): the horizon is finite and elapsed time is in the observation.
"""
import copy
import random
import time
from collections import deque

import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)


def mlp(n_in, n_out, hidden=128):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_out))


def run_episode(env, act, seed=None):
    s, _ = env.reset(seed=seed)
    done, ret = False, 0.0
    while not done:
        s, r, done, _, info = env.step(act(s))
        ret += r
    return ret, info


class DDQN:
    """Double DQN (van Hasselt et al., 2016) with experience replay and a periodically synced target net."""
    name = "DDQN"

    def __init__(self, obs_dim, n_actions, lr=3e-4, buffer=100_000, batch=64, sync=1000, eps_frac=0.5,
                 eps_end=0.05, gamma=1.0, train_freq=1, seed=0, **_):
        torch.manual_seed(seed)
        self.q = mlp(obs_dim, n_actions)
        self.tgt = copy.deepcopy(self.q)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buf = deque(maxlen=buffer)
        self.n_actions, self.batch, self.sync, self.gamma = n_actions, batch, sync, gamma
        self.train_freq = train_freq
        self.eps_frac, self.eps_end = eps_frac, eps_end
        self.rng = random.Random(seed)
        self.updates = 0

    @torch.no_grad()
    def act(self, s, greedy=True, eps=0.0):
        if not greedy and self.rng.random() < eps:
            return self.rng.randrange(self.n_actions)
        return int(self.q(torch.as_tensor(s)[None]).argmax(1).item())

    def _update(self):
        b = self.rng.sample(self.buf, self.batch)
        s, a, r, s2, d = map(np.array, zip(*b))
        s, s2 = torch.as_tensor(s), torch.as_tensor(s2)
        a, r, d = torch.as_tensor(a), torch.as_tensor(r, dtype=torch.float32), torch.as_tensor(d, dtype=torch.float32)
        q = self.q(s).gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            a2 = self.q(s2).argmax(1, keepdim=True)
            y = r + self.gamma * (1 - d) * self.tgt(s2).gather(1, a2).squeeze(1)
        loss = nn.functional.huber_loss(q, y)
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
            done = False
            while not done:
                eps = max(self.eps_end, 1 - (1 - self.eps_end) * t / (self.eps_frac * steps))
                a = self.act(s, greedy=False, eps=eps)
                s2, r, done, _, info = env.step(a)
                self.buf.append((s, a, r, s2, float(done)))
                s = s2
                t += 1
                if len(self.buf) >= 1000 and t % self.train_freq == 0:
                    self._update()
            ep += 1
            if log is not None:
                log.append((t, ep, info["is_bps"], time.time() - t0))

    def state_dict(self):
        return self.q.state_dict()

    def load_state_dict(self, sd):
        self.q.load_state_dict(sd)


class PPO:
    """Proximal Policy Optimization (Schulman et al., 2017), clipped objective, GAE advantages."""
    name = "PPO"

    def __init__(self, obs_dim, n_actions, lr=3e-4, rollout=2048, epochs=10, minibatch=256, clip=0.2,
                 ent=0.01, vf=0.5, gae=0.95, gamma=1.0, seed=0, **_):
        torch.manual_seed(seed)
        self.pi, self.v = mlp(obs_dim, n_actions), mlp(obs_dim, 1)
        self.opt = torch.optim.Adam(list(self.pi.parameters()) + list(self.v.parameters()), lr=lr)
        self.rollout, self.epochs, self.minibatch = rollout, epochs, minibatch
        self.clip, self.ent, self.vf, self.lam, self.gamma = clip, ent, vf, gae, gamma
        self.gen = torch.Generator().manual_seed(seed)

    @torch.no_grad()
    def act(self, s, greedy=True):
        logits = self.pi(torch.as_tensor(s)[None])
        if greedy:
            return int(logits.argmax(1).item())
        return int(torch.distributions.Categorical(logits=logits).sample().item())

    def train(self, env, steps, log=None, seed=0):
        rng = np.random.default_rng(seed)
        t, ep, t0 = 0, 0, time.time()
        s, _ = env.reset(seed=int(rng.integers(2 ** 31 - 1)))
        while t < steps:
            S, A, R, D, LP, V = [], [], [], [], [], []
            for _ in range(self.rollout):
                st = torch.as_tensor(s)[None]
                with torch.no_grad():
                    dist = torch.distributions.Categorical(logits=self.pi(st))
                    a = dist.sample()
                    lp, v = dist.log_prob(a), self.v(st)[0, 0]
                s2, r, done, _, info = env.step(int(a.item()))
                S.append(s); A.append(int(a.item())); R.append(r); D.append(done); LP.append(lp.item()); V.append(v.item())
                t += 1
                s = s2
                if done:
                    ep += 1
                    if log is not None:
                        log.append((t, ep, info["is_bps"], time.time() - t0))
                    s, _ = env.reset(seed=int(rng.integers(2 ** 31 - 1)))
            with torch.no_grad():
                v_last = self.v(torch.as_tensor(s)[None])[0, 0].item()
            adv = np.zeros(len(R), np.float32)
            g = 0.0
            for i in reversed(range(len(R))):
                nv = v_last if i == len(R) - 1 else V[i + 1]
                nonterm = 0.0 if D[i] else 1.0
                delta = R[i] + self.gamma * nv * nonterm - V[i]
                g = delta + self.gamma * self.lam * nonterm * g
                adv[i] = g
            ret = adv + np.array(V, np.float32)
            S_, A_ = torch.as_tensor(np.array(S)), torch.as_tensor(A)
            LP_, ADV, RET = torch.as_tensor(LP, dtype=torch.float32), torch.as_tensor(adv), torch.as_tensor(ret)
            ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
            n = len(R)
            for _ in range(self.epochs):
                perm = torch.randperm(n, generator=self.gen)
                for i in range(0, n, self.minibatch):
                    idx = perm[i:i + self.minibatch]
                    dist = torch.distributions.Categorical(logits=self.pi(S_[idx]))
                    ratio = torch.exp(dist.log_prob(A_[idx]) - LP_[idx])
                    pg = -torch.min(ratio * ADV[idx], torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * ADV[idx]).mean()
                    vl = ((self.v(S_[idx]).squeeze(1) - RET[idx]) ** 2).mean()
                    loss = pg + self.vf * vl - self.ent * dist.entropy().mean()
                    self.opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(list(self.pi.parameters()) + list(self.v.parameters()), 0.5)
                    self.opt.step()

    def state_dict(self):
        return {"pi": self.pi.state_dict(), "v": self.v.state_dict()}

    def load_state_dict(self, sd):
        self.pi.load_state_dict(sd["pi"])
        self.v.load_state_dict(sd["v"])


ALGOS = {"DDQN": DDQN, "PPO": PPO}
