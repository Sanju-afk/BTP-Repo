"""v2 training grid: every arm x seeds 10-19 on 'mixed', 250k steps, N jobs at a time. Skips finished runs.

    python3 experiments/run_grid_v2.py --procs 4
"""
import argparse
import os
import subprocess
import sys
from multiprocessing.pool import ThreadPool

ARMS = ["DDQN", "PPO", "QRDDQN-RN", "QRDDQN-CVaR25", "QRDDQN-CVaR10", "QRDDQN-THR25"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=250_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10, 20)))
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--runs", default="runs_v2")
    ap.add_argument("--procs", type=int, default=2)
    args = ap.parse_args()
    jobs = [(a, s) for s in args.seeds for a in args.arms]
    jobs.sort(key=lambda j: (j[0] == "PPO", j[1]))            # longer value-based jobs first
    todo = [j for j in jobs if not os.path.exists(os.path.join(args.runs, f"lob_{j[0]}_mixed_s{j[1]}", "model.pt"))]
    print(f"{len(todo)} of {len(jobs)} runs to do", flush=True)

    def run(j):
        cmd = [sys.executable, "experiments/train_v2.py", "--arm", j[0], "--seed", str(j[1]), "--steps", str(args.steps),
               "--runs", args.runs]
        out = subprocess.run(cmd, capture_output=True, text=True)
        print(out.stdout.strip() or out.stderr[-2000:], flush=True)

    with ThreadPool(args.procs) as p:
        list(p.imap_unordered(run, todo))


if __name__ == "__main__":
    main()
