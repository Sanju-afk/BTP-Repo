"""Run the training grid, two jobs at a time (one per core).

    python3 experiments/run_grid.py --steps 250000 --seeds 0 1 2 3 4

Grid (deck slide 12 plus the added ablations):
  lob  x {DDQN, PPO} x {L, M, H, mixed} x seeds        Experiments A-D and cross-regime generalisation
  lob  x {DDQN, PPO} x {mixed, oracle}  x seeds        value of observing the latent regime
  ac   x {DDQN, PPO} x {mixed}          x seeds        idealised Almgren-Chriss world (sub-question 4)
Skips runs that already have a model.
"""
import argparse
import os
import subprocess
import sys
from multiprocessing.pool import ThreadPool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.train import run_name   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=250_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--only", default=None, help="lob, oracle or ac")
    args = ap.parse_args()
    jobs = []
    for seed in args.seeds:
        for algo in ("DDQN", "PPO"):
            if args.only in (None, "lob"):
                jobs += [("lob", algo, tr, seed, False) for tr in ("mixed", "M", "L", "H")]
            if args.only in (None, "oracle"):
                jobs.append(("lob", algo, "mixed", seed, True))
            if args.only in (None, "ac"):
                jobs.append(("ac", algo, "mixed", seed, False))
    jobs.sort(key=lambda j: (j[1] != "DDQN", j[3]))           # long DDQN jobs first
    todo = [j for j in jobs if not os.path.exists(os.path.join(args.runs, run_name(j[0], j[1], j[2], j[3], j[4]), "model.pt"))]
    print(f"{len(todo)} of {len(jobs)} runs to do", flush=True)

    def run(j):
        world, algo, tr, seed, oracle = j
        cmd = [sys.executable, "experiments/train.py", "--world", world, "--algo", algo, "--train", tr, "--seed", str(seed),
               "--steps", str(args.steps), "--runs", args.runs] + (["--oracle"] if oracle else [])
        out = subprocess.run(cmd, capture_output=True, text=True)
        print(out.stdout.strip() or out.stderr[-2000:], flush=True)

    with ThreadPool(2) as p:
        list(p.imap_unordered(run, todo))


if __name__ == "__main__":
    main()
