"""Train one agent.

    python3 experiments/train.py --algo PPO --train M --seed 0 --steps 200000 [--world lob|ac] [--oracle]

Writes runs/<world>_<algo>_<train>[_oracle]_s<seed>/{model.pt, log.csv, config.json}.
--hp '{"lr": 1e-4}' overrides hyperparameters; by default the tuned ones in runs/tuned.json are used.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from liqexec.ac import ACIdealEnv          # noqa: E402
from liqexec.agents import ALGOS           # noqa: E402
from liqexec.env import N_ACTIONS, ExecutionEnv   # noqa: E402


def make_env(world, schedule, seed, oracle=False, calib="out/calib/regimes.json", ac="out/calib/ac_params.json", **kw):
    if world == "lob":
        return ExecutionEnv(calib, schedule=schedule, seed=seed, oracle=oracle, **kw)
    return ACIdealEnv(json.load(open(ac)), schedule=schedule, seed=seed, **kw)


def run_name(world, algo, train, seed, oracle=False, tag=""):
    return f"{world}_{algo}_{train}{'_oracle' if oracle else ''}{tag}_s{seed}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=list(ALGOS), required=True)
    ap.add_argument("--train", default="M", help="training schedule: L, M, H, HML, shock, markov, mixed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--world", choices=["lob", "ac"], default="lob")
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--hp", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--runs", default="runs")
    args = ap.parse_args()

    hp = {}
    tuned = os.path.join(args.runs, "tuned.json")
    if os.path.exists(tuned):
        hp = json.load(open(tuned)).get(args.algo, {})
    if args.hp:
        hp.update(json.loads(args.hp))
    env = make_env(args.world, args.train, args.seed, args.oracle)
    obs_dim = 13 + 3 * args.oracle
    agent = ALGOS[args.algo](obs_dim, N_ACTIONS, seed=args.seed, **hp)
    log = []
    agent.train(env, args.steps, log=log, seed=args.seed)
    d = os.path.join(args.runs, run_name(args.world, args.algo, args.train, args.seed, args.oracle, args.tag))
    os.makedirs(d, exist_ok=True)
    torch.save(agent.state_dict(), os.path.join(d, "model.pt"))
    np.savetxt(os.path.join(d, "log.csv"), np.array(log), delimiter=",", header="step,episode,is_bps,seconds", comments="")
    json.dump({**vars(args), "hp": hp, "obs_dim": obs_dim}, open(os.path.join(d, "config.json"), "w"), indent=1)
    last = np.array(log)[-200:, 2]
    print(f"{d}: {len(log)} episodes, {log[-1][3]:.0f} s, last-200 training IS {last.mean():.2f} bps")


if __name__ == "__main__":
    main()
