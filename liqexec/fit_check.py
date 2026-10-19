"""Fit of the *final* calibrated simulator, re-measured on many fresh seeds.

msm.py prints its fit on 4 to 10 fresh seeds, and it prints the dynamic moments right after the dynamic
stage, before `run_all.sh` re-runs the static stage (`--stage dyn+static`), which changes the dynamics a
little. This script re-measures every targeted moment with the final parameters and enough seeds that
simulation noise is negligible, so the realism table in the report comes from the simulator that was
actually used for training and testing, and can be regenerated with one command.

    python3 -m liqexec.fit_check out/calib/regimes.json            # writes out/validation/fit_final.json
"""
import argparse
import json

from .lobsim import Market
from .msm import DYN, MOMENTS, data_dyn_targets, dyn_moments, sim_moments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("calib")
    ap.add_argument("--out", default="out/validation/fit_final.json")
    ap.add_argument("--dyn-seeds", type=int, default=1000, help="seeds per regime for the dynamic moments (1 h each)")
    ap.add_argument("--static-seeds", type=int, default=200, help="seeds per regime for the static moments (30 min each)")
    args = ap.parse_args()
    p = json.load(open(args.calib))
    m = Market(p)
    out = {"static": {}, "dynamic": {}, "seeds": {"dynamic": args.dyn_seeds, "static": args.static_seeds}}

    print(f"== static moments (data vs final simulator, {args.static_seeds} fresh seeds per regime)")
    for r, name in enumerate(m.regimes):
        data = p["per_regime"][name]["data"]
        sim = sim_moments(m, r, seeds=tuple(range(9000, 9000 + args.static_seeds)))
        out["static"][name] = {k: {"data": data[k], "sim": sim[k], "rel_err": sim[k] / data[k] - 1}
                               for k in MOMENTS + ("mo_volume_per_s",)}
        for k in MOMENTS + ("mo_volume_per_s",):
            print(f"  {name} {k:22s} data {data[k]:8.2f}   sim {sim[k]:8.2f}   {sim[k] / data[k] - 1:+.0%}")

    print(f"== dynamic moments, pooled over regimes ({args.dyn_seeds} fresh seeds per regime)")
    tg = data_dyn_targets(p)
    sim = dyn_moments(m, seeds=tuple(range(5000, 5000 + args.dyn_seeds)))
    for k in DYN:
        t = (sim[k] - tg[k]) / tg["se"][k]
        out["dynamic"][k] = {"data": tg[k], "se": tg["se"][k], "sim": sim[k], "t": t}
        print(f"  {k:8s} data {tg[k]:7.3f} +- {tg['se'][k]:.3f}   sim {sim[k]:7.3f}   t = {t:+.1f}")
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
