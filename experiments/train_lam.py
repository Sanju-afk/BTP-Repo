"""Train DDQN (or PPO) with a given risk penalty lam' (v2: risk-averse DDQN). Phase-1 files are not touched.

    python3 experiments/train_lam.py --algo DDQN --lam 2 --seed 10 --steps 250000

Uses the Phase-1 agents and tuned hyperparameters (runs/tuned.json); only the reward's lam' changes.
Writes runs_v2/lob_<algo>_lam<lam>_s<seed>/{model.pt, log.csv, config.json}.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from liqexec.agents import ALGOS                 # noqa: E402
from liqexec.env import N_ACTIONS, ExecutionEnv  # noqa: E402


def run_name(algo, lam, seed):
    return f"lob_{algo}_lam{lam:g}_s{seed}"


def load_agent(run_dir):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    agent = ALGOS[cfg["algo"]](13, N_ACTIONS, seed=0, **cfg["hp"])
    agent.load_state_dict(torch.load(os.path.join(run_dir, "model.pt")))
    return agent, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=list(ALGOS), default="DDQN")
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=250_000)
    ap.add_argument("--runs", default="runs_v2")
    args = ap.parse_args()
    hp = json.load(open("runs/tuned.json"))[args.algo]
    env = ExecutionEnv(schedule="mixed", seed=args.seed, lam=args.lam)
    agent = ALGOS[args.algo](13, N_ACTIONS, seed=args.seed, **hp)
    log = []
    agent.train(env, args.steps, log=log, seed=args.seed)
    d = os.path.join(args.runs, run_name(args.algo, args.lam, args.seed))
    os.makedirs(d, exist_ok=True)
    torch.save(agent.state_dict(), os.path.join(d, "model.pt"))
    np.savetxt(os.path.join(d, "log.csv"), np.array(log), delimiter=",", header="step,episode,is_bps,seconds", comments="")
    json.dump({**vars(args), "hp": hp}, open(os.path.join(d, "config.json"), "w"), indent=1)
    print(f"{d}: {len(log)} episodes, {log[-1][3]:.0f} s, last-200 training IS {np.array(log)[-200:, 2].mean():.2f} bps",
          flush=True)


if __name__ == "__main__":
    main()
