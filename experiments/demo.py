"""Live demonstration: replay one episode for TWAP, Almgren-Chriss, DDQN and PPO on the same seed.

    python3 experiments/demo.py                       # the D2 shock episode shown in the report (Fig. 5.7)
    python3 experiments/demo.py --schedule HML --episode 42 --seed-run 3

Uses only saved artefacts (out/calib/*.json and runs/*/model.pt); runs in a few seconds on a laptop.
Because every policy gets the same episode seed, the background order flow is identical until the
policies' own orders change it (common random numbers). The default (test episode 7, i.e. seed
10,000,007, models of training seed 0) is the episode in the report and the Review-2 deck.

Prints, for each policy, the shortfall and the child orders it chose, and writes out/report/fig_demo.png.
"""
import argparse
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from evaluate import load_agent                  # noqa: E402
from liqexec.env import ExecutionEnv             # noqa: E402
from liqexec.policies import Baselines           # noqa: E402

TEST_SEED0 = 10_000_000                          # evaluate.py uses seeds 10,000,000 + i for test episode i
SWITCH = {"shock": [10], "HML": [6.67, 13.33]}


def replay(env, pol, seed):
    s, _ = env.reset(seed=seed)
    done, path, orders = False, [env.X], []
    while not done:
        a, ov = pol(env, s)
        s, r, done, _, info = env.step(a, override=ov)
        path.append(env.q)
    for k, qty, style in env.actions:
        orders.append((k, qty, style))
    return info, path, orders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", default="shock", help="L, M, H, HML, shock or markov")
    ap.add_argument("--episode", type=int, default=7, help="test episode index (seed = 10,000,000 + index)")
    ap.add_argument("--seed-run", type=int, default=0, help="which training seed's models to load")
    ap.add_argument("--out", default="out/report/fig_demo.png")
    args = ap.parse_args()

    env = ExecutionEnv(schedule=args.schedule)
    b = Baselines()
    pols = {"TWAP": lambda e, s: (None, b.twap(e)), "Almgren-Chriss": lambda e, s: (None, b.ac_static(e))}
    for algo in ("DDQN", "PPO"):
        pol, _ = load_agent(f"runs/lob_{algo}_mixed_s{args.seed_run}")
        pols[algo] = pol
    seed = TEST_SEED0 + args.episode

    res = {name: replay(env, pol, seed) for name, pol in pols.items()}
    print(f"schedule {args.schedule}, test episode {args.episode} (seed {seed}), models of training seed {args.seed_run}")
    for name, (info, path, orders) in res.items():
        print(f"\n{name:15s} IS {info['is_bps']:+6.2f} bps   sold before T {info['completion']:.0%}   "
              f"spread {info['c_spread']:+.2f}  impact {info['c_impact']:+.2f}  passive {info['c_passive']:+.2f}  "
              f"delay {info['c_delay']:+.2f}  fees {info['c_fees']:+.2f}")
        print("   child orders (step:shares/style): " + "  ".join(f"{k}:{q}/{st}" for k, q, st in orders if q > 0))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.1, 1]})
    for name, (info, path, _) in res.items():
        ax.plot(np.arange(len(path)), np.array(path) / env.X * 100, lw=2 if name in ("DDQN", "PPO") else 1.4,
                label=f"{name}: IS {info['is_bps']:.1f} bps")
    for s in SWITCH.get(args.schedule, []):
        ax.axvline(s, color="grey", ls=":", lw=1)
    ax.set_xlabel("decision step (30 s)")
    ax.set_ylabel("inventory remaining (% of X)")
    ax.set_title(f"One {args.schedule} episode, same seed for all", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    colours = {"MO": "#333333", "JOIN": "#6c5ce7", "IMPROVE": "#e1a100"}
    for row, name in enumerate(("DDQN", "PPO")):
        for k, q, st in res[name][2]:
            if q > 0:
                ax2.bar(k, q / env.slice * 0.3, bottom=row, color=colours[st], width=0.8)
    ax2.set_yticks([0.45, 1.45], ["DDQN", "PPO"])
    ax2.set_xlabel("decision step (30 s)   (bar height = size submitted)")
    ax2.set_title("Child orders chosen", fontsize=10)
    for st, c in colours.items():
        ax2.bar(0, 0, color=c, label={"MO": "market order", "JOIN": "limit, join queue", "IMPROVE": "limit, improve 1 tick"}[st])
    ax2.legend(fontsize=8, frameon=False, loc="center right")
    ax2.set_xlim(-1, env.N)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"\nfigure: {args.out}")


if __name__ == "__main__":
    main()
