"""Evaluate the v2 risk-penalty sweep on the Phase-1 test episodes (common random numbers).

    python3 experiments/evaluate_lam.py --episodes 1000 --out out_v2/eval --procs 4

For every lam' and setting: the agents trained with that lam' (runs_v2/lob_*_lam<lam>_s*), and the
Almgren-Chriss baselines solved for the same lam' (named AC@<lam>, AC-adaptive@<lam>). The lam'-free baselines
(TWAP, TWAP-LO, VWAP, POV) are evaluated once, at lam' = 0.5. Same columns as experiments/evaluate.py plus
lam, forced (share of X sold by the forced order at T) and inv_k10 (inventory at the D2 shock time / X).
One CSV per (setting, lam').
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
from run_grid_lam import LAMS                    # noqa: E402
from train_lam import load_agent                 # noqa: E402

torch.set_num_threads(1)
LAM_FREE = ["TWAP", "TWAP-LO", "VWAP", "POV"]


def policies(runs, lam):
    out = {}
    for name, f in Baselines().all().items():
        if name in LAM_FREE and lam != 0.5:
            continue
        key = name if name in LAM_FREE else f"{name}@{lam:g}"
        out[key] = (lambda f: (lambda env, s: (None, f(env))))(f)
    for d in sorted(glob.glob(os.path.join(runs, f"lob_*_lam{lam:g}_s*"))):
        if os.path.exists(os.path.join(d, "model.pt")):
            agent, _ = load_agent(d)
            out[os.path.basename(d)] = (lambda ag: (lambda env, s: (ag.act(s, greedy=True), None)))(agent)
    return out


def run_job(args):
    cond, lam, episodes, runs, out_dir, calib, latency, fee_mult = args
    env = ExecutionEnv(calib, schedule=cond, lam=lam, latency=latency, taker_fee=0.003 * fee_mult,
                       maker_rebate=0.002 * fee_mult)
    rows = []
    for name, pol in policies(runs, lam).items():
        for i in range(episodes):
            s, _ = env.reset(seed=10_000_000 + i)
            done, ret, path = False, 0.0, [env.X]
            while not done:
                a, ov = pol(env, s)
                s, r, done, _, info = env.step(a, override=ov)
                ret += r
                path.append(env.q)
            styles = [st for _, q, st in env.actions if q > 0]
            rows.append({"policy": name, "condition": cond, "lam": lam, "episode": i, "is_bps": info["is_bps"],
                         "objective": -ret, "completion": info["completion"], "schedule": info["schedule"],
                         "limit_share": np.mean([st != "MO" for st in styles]) if styles else 0.0,
                         **{k: v for k, v in info.items() if k.startswith("c_")},
                         "forced": 1 - info["completion"], "inv_k10": path[10] / env.X,
                         "path": " ".join(str(int(x)) for x in path)})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"{cond}_lam{lam:g}.csv"), index=False)
    return f"{cond} lam {lam:g}: {len(rows)} episodes"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=["L", "M", "H", "HML", "shock", "markov"])
    ap.add_argument("--lams", type=float, nargs="+", default=LAMS)
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--runs", default="runs_v2")
    ap.add_argument("--out", default="out_v2/eval")
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--calib", default="out/calib/regimes.json")
    ap.add_argument("--latency", type=float, default=0.05)
    ap.add_argument("--fee-mult", type=float, default=1.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    jobs = [(c, lam, args.episodes, args.runs, args.out, args.calib, args.latency, args.fee_mult)
            for lam in args.lams for c in args.conditions]
    jobs.sort(key=lambda j: -(j[1] == 0.5))
    with Pool(args.procs) as p:
        for msg in p.imap_unordered(run_job, jobs):
            print("done", msg, flush=True)


if __name__ == "__main__":
    main()
