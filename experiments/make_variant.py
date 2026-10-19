"""Simulator variants for the sensitivity analysis (RESULTS.md section 8).

    python3 experiments/make_variant.py --kappa-mult 2 --out out/calib/regimes_fastres.json

"Resilience x2" doubles the tilt kappa toward the latent fundamental price (liqexec/lobsim.py), so impact
reverts faster, closer to the data's 2 to 5 minute reversion. Every other parameter is unchanged.
Latency and fee variants need no new calibration file; they are flags of experiments/evaluate.py.
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="out/calib/regimes.json")
    ap.add_argument("--kappa-mult", type=float, default=2.0)
    ap.add_argument("--out", default="out/calib/regimes_fastres.json")
    a = ap.parse_args()
    p = json.load(open(a.calib))
    p["dyn"]["kappa"] *= a.kappa_mult
    p["variant"] = f"fast resilience: kappa x{a.kappa_mult:g} (sensitivity analysis)"
    json.dump(p, open(a.out, "w"))
    print(f"kappa {p['dyn']['kappa'] / a.kappa_mult:.5f} -> {p['dyn']['kappa']:.5f}; written to {a.out}")


if __name__ == "__main__":
    main()
