"""Evaluate the v2 arms (runs_v2/) and the baselines on the Phase-1 test episodes (common random numbers).

    python3 experiments/evaluate_v2.py --episodes 1000 --out out_v2/eval --procs 4

Same episode seeds (10,000,000 + i), same environment, same CSV columns as experiments/evaluate.py, so the
Phase-1 bootstrap code applies unchanged. Agents with an episode state (the threshold-CVaR arm) get
begin_episode() at reset and observe(r) after every reward; for the other arms both are no-ops.
Extra columns: forced = share of X sold by the forced market order at T; inv_k10 = inventory at step 10
(the D2 shock time) / X.
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from liqexec.env import ExecutionEnv             # noqa: E402
from liqexec.policies import Baselines           # noqa: E402
from train_v2 import load_agent                  # noqa: E402

torch.set_num_threads(1)


class AgentPolicy:
    def __init__(self, agent):
        self.agent = agent

    def begin(self, s):
        if hasattr(self.agent, "begin_episode"):
            self.agent.begin_episode(s)

    def __call__(self, env, s):
        return self.agent.act(s, greedy=True), None

    def observe(self, r):
        if hasattr(self.agent, "observe"):
            self.agent.observe(r)


class BaselinePolicy:
    def __init__(self, f):
        self.f = f

    def begin(self, s):
        pass

    def __call__(self, env, s):
        return None, self.f(env)

    def observe(self, r):
        pass


def policies(runs, pattern, baselines):
    out = {}
    if baselines:
        out = {name: BaselinePolicy(f) for name, f in Baselines().all().items()}
    for d in sorted(glob.glob(os.path.join(runs, pattern))):
        if os.path.exists(os.path.join(d, "model.pt")):
            out[os.path.basename(d)] = AgentPolicy(load_agent(d))
    return out


def run_condition(args):
    cond, episodes, runs, pattern, out_dir, calib, latency, fee_mult, baselines = args
    pols = policies(runs, pattern, baselines)
    env = ExecutionEnv(calib, schedule=cond, latency=latency, taker_fee=0.003 * fee_mult, maker_rebate=0.002 * fee_mult)
    rows = []
    for name, pol in pols.items():
        for i in range(episodes):
            s, _ = env.reset(seed=10_000_000 + i)
            pol.begin(s)
            done, ret, path = False, 0.0, [env.X]
            while not done:
                a, ov = pol(env, s)
                s, r, done, _, info = env.step(a, override=ov)
                pol.observe(r)
                ret += r
                path.append(env.q)
            styles = [st for _, q, st in env.actions if q > 0]
            rows.append({"policy": name, "condition": cond, "episode": i, "is_bps": info["is_bps"], "objective": -ret,
                         "completion": info["completion"], "schedule": info["schedule"],
                         "limit_share": np.mean([st != "MO" for st in styles]) if styles else 0.0,
                         **{k: v for k, v in info.items() if k.startswith("c_")},
                         "forced": 1 - info["completion"], "inv_k10": path[10] / env.X,
                         "path": " ".join(str(int(x)) for x in path)})
        print(f"  {cond:7s} {name:32s} IS {np.mean([r['is_bps'] for r in rows if r['policy'] == name]):6.2f}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"{cond}.csv"), index=False)
    return cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=["L", "M", "H", "HML", "shock", "markov"])
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--runs", default="runs_v2")
    ap.add_argument("--pattern", default="lob_*")
    ap.add_argument("--out", default="out_v2/eval")
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--calib", default="out/calib/regimes.json")
    ap.add_argument("--latency", type=float, default=0.05)
    ap.add_argument("--fee-mult", type=float, default=1.0)
    ap.add_argument("--no-baselines", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    jobs = [(c, args.episodes, args.runs, args.pattern, args.out, args.calib, args.latency, args.fee_mult,
             not args.no_baselines) for c in args.conditions]
    with Pool(args.procs) as p:
        for c in p.imap_unordered(run_condition, jobs):
            print("done", c, flush=True)


if __name__ == "__main__":
    main()
