"""Invariants of the simulator and the environment.  python3 -m pytest -q tests/  (or python3 tests/test_sim.py)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from liqexec.env import N_ACTIONS, ExecutionEnv   # noqa: E402
from liqexec.lobsim import NA, NB, Market         # noqa: E402

CALIB = os.path.join(os.path.dirname(__file__), "..", "out", "calib", "regimes.json")


def check_book(m):
    bk, st = m.bk, m.st
    assert st[NB] == 10 and st[NA] == 10, "both sides keep K levels"
    assert np.all(np.diff(bk[0]) < 0), "bid prices strictly decreasing"
    assert np.all(np.diff(bk[2]) > 0), "ask prices strictly increasing"
    assert np.all(bk[1] > 0) and np.all(bk[3] > 0), "every level has shares"
    assert m.best_bid() < m.best_ask(), "book not crossed"


def test_book_invariants_with_agent():
    m = Market(CALIB)
    rng = np.random.default_rng(0)
    for r in range(3):
        m.reset(r, seed=r)
        for i in range(600):
            m.run_until(m.t + 1.0)
            check_book(m)
            u = rng.random()
            if u < 0.05:
                m.agent_market(int(rng.integers(1, 800)))
                check_book(m)
            elif u < 0.15:
                m.cancel_agent()
                m.agent_limit(int(rng.integers(1, 800)), improve=bool(rng.integers(2)))
                if m.st[2]:
                    assert m.st[3] > m.best_bid(), "agent's limit sell rests above the best bid"
            if i % 50 == 0:
                m.set_regime(int(rng.integers(3)), shock_frac=0.3 * rng.random())
                check_book(m)


def test_agent_fills_bounded():
    m = Market(CALIB)
    m.reset(1, seed=5)
    m.agent_limit(300, improve=True)
    m.run_until(m.t + 120)
    left = m.cancel_agent()
    f = m.take_fills()
    assert f[:, 0].sum() + left == 300 and np.all(f[:, 2] == 1)
    m.agent_market(1000)
    f = m.take_fills()
    assert f[:, 0].sum() == 1000 and np.all(f[:, 2] == 0)
    assert np.all(np.diff(f[:, 1]) <= 0), "a market sell fills at non-increasing prices"


def test_reward_identity_and_completion():
    for sched in ("L", "HML", "shock", "markov"):
        env = ExecutionEnv(CALIB, schedule=sched, lam=0.7)
        rng = np.random.default_rng(1)
        for ep in range(20):
            env.reset(seed=ep)
            done, ret, pen = False, 0.0, 0.0
            while not done:
                _, r, done, _, info = env.step(int(rng.integers(N_ACTIONS)))
                ret += r
                pen += env.lam * (env.q / env.X) ** 2
            assert env.q == 0, "forced completion sells everything"
            assert abs(-ret - pen - info["is_bps"]) < 1e-6, "sum of rewards = -IS - risk penalty"
            parts = sum(info[k] for k in ("c_delay", "c_spread", "c_impact", "c_passive", "c_fees"))
            assert abs(parts - info["is_bps"]) < 1e-6, "cost decomposition adds up to the shortfall"


def test_determinism():
    out = []
    for _ in range(2):
        env = ExecutionEnv(CALIB, schedule="markov")
        env.reset(seed=42)
        done, rs = False, []
        while not done:
            _, r, done, _, _ = env.step(4)
            rs.append(r)
        out.append(rs)
    assert out[0] == out[1], "same seed, same actions -> identical episode"


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
