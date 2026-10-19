"""Equal-budget tuning for the two tuned QR-DDQN arms, with the Phase-1 DDQN grid and protocol.

Same 4 configurations as DDQN in experiments/tune.py (lr {1e-4, 5e-4} x (batch, train_freq) {(64, 1),
(128, 4)}), 150k steps on 'mixed' with seed 100, scored on the same validation episodes (seeds 20,000,000+)
in all six conditions; selection by mean risk-adjusted objective. alpha is fixed (0.25) for the CVaR arm.
Writes runs_v2/tuned.json and runs_v2/tuning.csv.

    python3 experiments/tune_v2.py --steps 150000 --val-episodes 150 --procs 4
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
from liqexec.env import N_ACTIONS, ExecutionEnv  # noqa: E402
from liqexec.qr_ddqn import ARMS, QRDDQN         # noqa: E402

torch.set_num_threads(1)
TUNED_ARMS = ["QRDDQN-RN", "QRDDQN-CVaR25"]
GRID = [dict(lr=lr, batch=b, train_freq=f) for lr, (b, f) in itertools.product([1e-4, 5e-4], [(64, 1), (128, 4)])]
CONDS = ["L", "M", "H", "HML", "shock", "markov"]


def job(args):
    arm, i, hp, steps, val = args
    env = ExecutionEnv(schedule="mixed", seed=100)
    agent = QRDDQN(13, N_ACTIONS, seed=100, **{**hp, **ARMS[arm]})
    log = []
    agent.train(env, steps, log=log, seed=100)
    res = {}
    for c in CONDS:
        e = ExecutionEnv(schedule=c)
        objs = []
        for k in range(val):
            s, _ = e.reset(seed=20_000_000 + k)
            agent.begin_episode(s)
            done, ret = False, 0.0
            while not done:
                s, r, done, _, info = e.step(agent.act(s, greedy=True))
                agent.observe(r)
                ret += r
            objs.append(-ret)
        res[c] = float(np.mean(objs))
    out = {"arm": arm, "config": i, **hp, **{f"obj_{c}": v for c, v in res.items()},
           "obj_mean": float(np.mean(list(res.values()))), "train_seconds": log[-1][3]}
    print(out, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150_000)
    ap.add_argument("--val-episodes", type=int, default=150)
    ap.add_argument("--runs", default="runs_v2")
    ap.add_argument("--procs", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(args.runs, exist_ok=True)
    jobs = [(a, i, hp, args.steps, args.val_episodes) for a in TUNED_ARMS for i, hp in enumerate(GRID)]
    jobs.sort(key=lambda j: -(j[2]["train_freq"] == 1))
    with Pool(args.procs) as p:
        rows = list(p.imap_unordered(job, jobs))
    df = pd.DataFrame(rows).sort_values(["arm", "obj_mean"])
    df.to_csv(os.path.join(args.runs, "tuning.csv"), index=False)
    best = {a: GRID[int(g.iloc[0]["config"])] for a, g in df.groupby("arm")}
    json.dump(best, open(os.path.join(args.runs, "tuned.json"), "w"), indent=1)
    print(df.to_string())
    print("selected:", best)


if __name__ == "__main__":
    main()
