"""Equal-budget hyperparameter selection (Henderson et al., 2018; Tonkin et al., 2025).

Each algorithm gets the same number of configurations (4) and the same training budget per configuration,
trained on the 'mixed' liquidity distribution with seed 100 and scored on validation episodes (seeds
20,000,000+, disjoint from test seeds) in every test condition. Selection criterion: mean risk-adjusted
objective (the quantity both agents optimise). Writes runs/tuned.json and runs/tuning.csv.

    python3 experiments/tune.py --steps 150000 --val-episodes 150
"""
import argparse
import itertools
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from liqexec.agents import ALGOS                 # noqa: E402
from liqexec.env import N_ACTIONS, ExecutionEnv  # noqa: E402

torch.set_num_threads(1)
GRID = {
    "DDQN": [dict(lr=lr, batch=b, train_freq=f) for lr, (b, f) in itertools.product([1e-4, 5e-4], [(64, 1), (128, 4)])],
    "PPO": [dict(lr=lr, ent=e) for lr, e in itertools.product([1e-4, 3e-4], [0.0, 0.01])],
}
CONDS = ["L", "M", "H", "HML", "shock", "markov"]


def job(args):
    algo, i, hp, steps, val = args
    env = ExecutionEnv(schedule="mixed", seed=100)
    agent = ALGOS[algo](13, N_ACTIONS, seed=100, **hp)
    log = []
    agent.train(env, steps, log=log, seed=100)
    res = {}
    for c in CONDS:
        e = ExecutionEnv(schedule=c)
        objs = []
        for k in range(val):
            s, _ = e.reset(seed=20_000_000 + k)
            done, ret = False, 0.0
            while not done:
                s, r, done, _, info = e.step(agent.act(s, greedy=True))
                ret += r
            objs.append(-ret)
        res[c] = float(np.mean(objs))
    out = {"algo": algo, "config": i, **hp, **{f"obj_{c}": v for c, v in res.items()},
           "obj_mean": float(np.mean(list(res.values()))), "train_seconds": log[-1][3]}
    print(out, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150_000)
    ap.add_argument("--val-episodes", type=int, default=150)
    ap.add_argument("--runs", default="runs")
    args = ap.parse_args()
    os.makedirs(args.runs, exist_ok=True)
    jobs = [(a, i, hp, args.steps, args.val_episodes) for a, g in GRID.items() for i, hp in enumerate(g)]
    jobs.sort(key=lambda j: -(j[0] == "DDQN" and j[2]["train_freq"] == 1))     # longest first
    with Pool(2) as p:
        rows = list(p.imap_unordered(job, jobs))
    df = pd.DataFrame(rows).sort_values(["algo", "obj_mean"])
    df.to_csv(os.path.join(args.runs, "tuning.csv"), index=False)
    best = {a: {k: v for k, v in GRID[a][int(g.iloc[0]["config"])].items()} for a, g in df.groupby("algo")}
    json.dump(best, open(os.path.join(args.runs, "tuned.json"), "w"), indent=1)
    print(df.to_string())
    print("selected:", best)


if __name__ == "__main__":
    main()
