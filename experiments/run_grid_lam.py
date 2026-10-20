"""v2 grid: DDQN at every lam' and PPO at lam' = 0.5, seeds 10-19, N jobs at a time. Skips finished runs.

    python3 experiments/run_grid_lam.py --procs 4
"""
import argparse
import os
import subprocess
import sys
from multiprocessing.pool import ThreadPool

sys.path.insert(0, os.path.dirname(__file__))
from train_lam import run_name   # noqa: E402

LAMS = [0, 0.5, 1, 2, 4, 8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=250_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10, 20)))
    ap.add_argument("--runs", default="runs_v2")
    ap.add_argument("--procs", type=int, default=2)
    args = ap.parse_args()
    jobs = [("DDQN", lam, s) for s in args.seeds for lam in LAMS] + [("PPO", 0.5, s) for s in args.seeds]
    todo = [j for j in jobs if not os.path.exists(os.path.join(args.runs, run_name(*j), "model.pt"))]
    print(f"{len(todo)} of {len(jobs)} runs to do", flush=True)

    def run(j):
        cmd = [sys.executable, "experiments/train_lam.py", "--algo", j[0], "--lam", str(j[1]), "--seed", str(j[2]),
               "--steps", str(args.steps), "--runs", args.runs]
        out = subprocess.run(cmd, capture_output=True, text=True)
        print(out.stdout.strip() or out.stderr[-2000:], flush=True)

    with ThreadPool(args.procs) as p:
        list(p.imap_unordered(run, todo))


if __name__ == "__main__":
    main()
