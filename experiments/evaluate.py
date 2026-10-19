"""Evaluate baselines and trained agents on common episodes (paired by seed).

    python3 experiments/evaluate.py --conditions L M H HML shock markov --episodes 500 --runs runs --out out/eval

For every policy and test condition, the same E episode seeds are used, so the background order flow
is identical until the policies' own orders change it (common random numbers): differences between
policies are paired. One CSV row per (policy, episode): shortfall, completion, cost decomposition, the
risk-penalised objective, and the inventory path.
"""
import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from liqexec.ac import ACIdealEnv                # noqa: E402
from liqexec.agents import ALGOS                 # noqa: E402
from liqexec.env import N_ACTIONS, ExecutionEnv  # noqa: E402
from liqexec.policies import Baselines           # noqa: E402

torch.set_num_threads(1)
AC_VISIBLE = [0, 1, 10]          # features an agent trained in the idealised AC world has seen


def load_agent(run_dir):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    agent = ALGOS[cfg["algo"]](cfg["obs_dim"], N_ACTIONS, seed=0, **cfg["hp"])
    agent.load_state_dict(torch.load(os.path.join(run_dir, "model.pt")))
    mask = None
    if cfg["world"] == "ac":
        mask = np.zeros(13, np.float32)
        mask[AC_VISIBLE] = 1
    oracle = cfg.get("oracle", False)

    def policy(env, s):
        if mask is not None:
            s = s[:13] * mask
        return agent.act(s, greedy=True), None
    return policy, oracle


def policies(runs, pattern):
    b = Baselines()
    out = {name: ((lambda f: (lambda env, s: (None, f(env))))(f), False) for name, f in b.all().items()}
    for d in sorted(glob.glob(os.path.join(runs, pattern))):
        if os.path.exists(os.path.join(d, "model.pt")):
            out[os.path.basename(d)] = load_agent(d)
    return out


def run_condition(args):
    cond, episodes, runs, pattern, out_dir, X, lam, world, calib, latency, fee_mult = args
    pols = policies(runs, pattern)
    if world == "ac":
        acp = json.load(open("out/calib/ac_params.json"))
        envs = {o: ACIdealEnv(acp, schedule=cond, X=X, lam=lam, taker_fee=0.003 * fee_mult) for o in (False, True)}
    else:
        envs = {o: ExecutionEnv(calib, schedule=cond, oracle=o, X=X, lam=lam, latency=latency,
                                taker_fee=0.003 * fee_mult, maker_rebate=0.002 * fee_mult) for o in (False, True)}
    rows = []
    for name, (pol, oracle) in pols.items():
        env = envs[oracle]
        for i in range(episodes):
            s, _ = env.reset(seed=10_000_000 + i)
            done, ret, path = False, 0.0, [env.X]
            while not done:
                a, ov = pol(env, s)
                s, r, done, _, info = env.step(a, override=ov)
                ret += r
                path.append(env.q)
            styles = [st for _, q, st in env.actions if q > 0]
            rows.append({"policy": name, "condition": cond, "episode": i, "is_bps": info["is_bps"], "objective": -ret,
                         "completion": info["completion"], "schedule": info["schedule"],
                         "limit_share": np.mean([st != "MO" for st in styles]) if styles else 0.0,
                         **{k: v for k, v in info.items() if k.startswith("c_")},
                         "path": " ".join(str(int(x)) for x in path)})
        print(f"  {cond:7s} {name:28s} IS {np.mean([r['is_bps'] for r in rows if r['policy'] == name]):6.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, f"{cond}.csv"), index=False)
    return cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=["L", "M", "H", "HML", "shock", "markov"])
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--pattern", default="lob_*")
    ap.add_argument("--out", default="out/eval")
    ap.add_argument("--X", type=int, default=6000)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--world", choices=["lob", "ac"], default="lob")
    ap.add_argument("--calib", default="out/calib/regimes.json", help="simulator variant for sensitivity tests")
    ap.add_argument("--latency", type=float, default=0.05)
    ap.add_argument("--fee-mult", type=float, default=1.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    jobs = [(c, args.episodes, args.runs, args.pattern, args.out, args.X, args.lam, args.world, args.calib, args.latency,
             args.fee_mult) for c in args.conditions]
    with Pool(args.procs) as p:
        for c in p.imap_unordered(run_condition, jobs):
            print("done", c, flush=True)


if __name__ == "__main__":
    main()
