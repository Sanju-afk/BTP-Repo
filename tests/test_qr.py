"""Tests for the v2 QR-DDQN agent.  python3 tests/test_qr.py  (or python3 -m pytest -q tests/)"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from liqexec.env import N_ACTIONS, ExecutionEnv                                    # noqa: E402
from liqexec.qr_ddqn import (ARMS, QRDDQN, cvar_lower, quantile_huber_loss,        # noqa: E402
                             risk_score, var_lower)

CALIB = os.path.join(os.path.dirname(__file__), "..", "out", "calib", "regimes.json")


def test_quantile_loss_shape_and_sign():
    K, B = 8, 5
    taus = (torch.arange(K, dtype=torch.float32) + 0.5) / K
    pred = torch.zeros(B, K, requires_grad=True)
    loss = quantile_huber_loss(pred, torch.ones(B, 16), taus)
    assert loss.dim() == 0 and loss.item() > 0
    loss.backward()
    assert pred.grad.shape == (B, K)
    assert torch.all(pred.grad < 0), "targets above every quantile: gradient descent raises all quantiles"
    pred2 = torch.zeros(B, K, requires_grad=True)
    quantile_huber_loss(pred2, -torch.ones(B, 16), taus).backward()
    assert torch.all(pred2.grad > 0), "targets below every quantile: gradient descent lowers all quantiles"
    # asymmetric weights: a positive error is weighted by tau, so high quantiles move up faster
    g = pred.grad[0]
    assert torch.all(g[1:] < g[:-1]), "weight |tau - 1{u<0}| grows with tau for u > 0"


def test_quantile_loss_recovers_quantiles():
    torch.manual_seed(0)
    K = 10
    taus = (torch.arange(K, dtype=torch.float32) + 0.5) / K
    theta = torch.zeros(1, K, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=0.05)
    for _ in range(3000):
        y = torch.randn(1, 64)
        loss = quantile_huber_loss(theta, y, taus, kappa=0.1)
        opt.zero_grad(); loss.backward(); opt.step()
    true = torch.distributions.Normal(0, 1).icdf(taus)
    assert torch.max((theta[0] - true).abs()) < 0.15, (theta, true)


def test_cvar_picks_best_lower_tail():
    # action 0: higher mean but a bad tail; action 1: lower mean, safe tail (returns, higher is better)
    z = torch.tensor([[[-10.0, -9.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
                       [-1.0, -0.5, 0.0, 0.5, 1.0, 1.0, 1.5, 2.0, 2.0, 2.5]]])
    assert risk_score(z, "mean", 0.25).argmax(-1).item() == 0
    assert risk_score(z, "cvar", 0.25).argmax(-1).item() == 1
    assert risk_score(z, "cvar", 0.10).argmax(-1).item() == 1
    assert abs(cvar_lower(z[0, 0], 0.2).item() - (-9.5)) < 1e-6
    assert abs(var_lower(z[0, 0], 0.2).item() - (-9.0)) < 1e-6
    # unsorted quantiles are sorted before the tail is taken
    zs = z[..., torch.randperm(10)]
    assert torch.allclose(cvar_lower(zs, 0.3), cvar_lower(z, 0.3))
    # threshold rule: with a low threshold only the downside below it matters
    c = torch.tensor([[-5.0, -5.0]])
    assert risk_score(z, "threshold", 0.25, c).argmax(-1).item() == 1


def test_agent_shapes_and_threshold_bookkeeping():
    for arm, kw in ARMS.items():
        ag = QRDDQN(13, N_ACTIONS, seed=0, **kw)
        x = torch.zeros(4, ag.in_dim)
        assert ag.quantiles(x).shape == (4, N_ACTIONS, ag.K)
        s = np.zeros(13, np.float32)
        ag.begin_episode(s)
        c0 = ag.c
        assert 0 <= ag.act(s) < N_ACTIONS
        ag.observe(-1.5)
        assert ag.c == (c0 + 1.5 if kw["risk"] == "threshold" else 0.0)


def test_reward_identity_with_qr_agent_and_update_runs():
    env = ExecutionEnv(CALIB, schedule="shock")
    for arm, kw in ARMS.items():
        ag = QRDDQN(13, N_ACTIONS, seed=1, batch=16, **kw)
        log = []
        ag.train(env, 1200, log=log, seed=1)            # passes the 1,000-transition warm-up, so updates run
        assert ag.updates > 0
        s, _ = env.reset(seed=3)
        ag.begin_episode(s)
        done, ret, pen = False, 0.0, 0.0
        while not done:
            s, r, done, _, info = env.step(ag.act(s))
            ag.observe(r)
            ret += r
            pen += env.lam * (env.q / env.X) ** 2
        assert env.q == 0
        assert abs(-ret - pen - info["is_bps"]) < 1e-6, "sum of rewards = -IS - risk penalty"


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
