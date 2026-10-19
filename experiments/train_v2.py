"""Train one agent of the v2 study (CVaR distributional RL extension). Phase-1 files are not touched.

    python3 experiments/train_v2.py --arm QRDDQN-CVaR25 --seed 10 --steps 250000

Arms: DDQN and PPO (Phase-1 classes and tuned hyperparameters, runs/tuned.json) and the QR-DDQN arms of
liqexec/qr_ddqn.py (hyperparameters from runs_v2/tuned.json). Always trained on the 'mixed' schedule.
Writes runs_v2/lob_<arm>_mixed_s<seed>/{model.pt, log.csv, config.json}.
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
from liqexec.qr_ddqn import ARMS, QRDDQN         # noqa: E402

# Arms whose hyperparameters are not tuned separately borrow those of a tuned arm (pre-registered).
HP_FROM = {"QRDDQN-CVaR10": "QRDDQN-CVaR25", "QRDDQN-THR25": "QRDDQN-CVaR25", "QRDDQN-RN4": "QRDDQN-CVaR25"}


def make_agent(arm, seed, hp):
    if arm in ALGOS:
        return ALGOS[arm](13, N_ACTIONS, seed=seed, **hp)
    return QRDDQN(13, N_ACTIONS, seed=seed, **{**hp, **ARMS[arm]})


def arm_hp(arm, runs_v2="runs_v2", runs_v1="runs"):
    if arm in ALGOS:
        return json.load(open(os.path.join(runs_v1, "tuned.json")))[arm]
    return json.load(open(os.path.join(runs_v2, "tuned.json")))[HP_FROM.get(arm, arm)]


def load_agent(run_dir):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    agent = make_agent(cfg["arm"], 0, cfg["hp"])
    agent.load_state_dict(torch.load(os.path.join(run_dir, "model.pt")))
    return agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ALGOS) + list(ARMS), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=250_000)
    ap.add_argument("--runs", default="runs_v2")
    args = ap.parse_args()
    hp = arm_hp(args.arm, args.runs)
    env = ExecutionEnv(schedule="mixed", seed=args.seed)
    agent = make_agent(args.arm, args.seed, hp)
    log = []
    agent.train(env, args.steps, log=log, seed=args.seed)
    d = os.path.join(args.runs, f"lob_{args.arm}_mixed_s{args.seed}")
    os.makedirs(d, exist_ok=True)
    torch.save(agent.state_dict(), os.path.join(d, "model.pt"))
    np.savetxt(os.path.join(d, "log.csv"), np.array(log), delimiter=",", header="step,episode,is_bps,seconds", comments="")
    json.dump({**vars(args), "hp": hp}, open(os.path.join(d, "config.json"), "w"), indent=1)
    last = np.array(log)[-200:, 2]
    print(f"{d}: {len(log)} episodes, {log[-1][3]:.0f} s, last-200 training IS {last.mean():.2f} bps", flush=True)


if __name__ == "__main__":
    main()
