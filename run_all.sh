#!/usr/bin/env bash
# Full pipeline, in order. About 5 hours on 2 cores: training about 3.2 h (6.4 core-hours, runs_grid.log),
# tuning about 30 min, calibration and validation about 20 min, evaluation and report about 35 min.
# To check the reported numbers without retraining, run only steps 8-10 (see README, "Reproduce").
#   ./run_all.sh            everything
#   STEPS=100000 ./run_all.sh   shorter training for a quick pass
set -euo pipefail
cd "$(dirname "$0")"
MSG=${MSG:-data/lobster/AAPL_2012-06-21_34200000_57600000_message_10.csv}
STEPS=${STEPS:-250000}
EPISODES=${EPISODES:-1000}

echo "== 1. calibration from LOBSTER data (regimes, queue-reactive rates, Hawkes, dynamics)"
python3 -m liqexec.calibrate "$MSG" --out out/calib
echo "== 2. method of simulated moments (resilience stage, then static stage)"
python3 -m liqexec.msm out/calib/regimes.json --stage dyn+static --iters 300 | tee out/calib/msm.log
echo "== 3. simulator validation (untargeted distributions, metaorder impact)"
mkdir -p out/validation
python3 -m liqexec.validate_sim out/calib/regimes.json "$MSG" --seeds 3000 | tee out/validation/validate.log
python3 -m liqexec.fit_check out/calib/regimes.json | tee out/validation/fit_final.log
echo "== 4. Almgren-Chriss parameters inside each regime, and the fast-resilience variant"
python3 -m liqexec.ac --episodes 300
python3 experiments/make_variant.py --kappa-mult 2 --out out/calib/regimes_fastres.json
echo "== 5. tests"
python3 tests/test_sim.py
echo "== 6. equal-budget tuning"
python3 experiments/tune.py --steps 150000 --val-episodes 150
echo "== 7. training grid"
python3 experiments/run_grid.py --steps "$STEPS" --seeds 0 1 2 3 4
echo "== 8. evaluation"
python3 experiments/evaluate.py --episodes "$EPISODES" --pattern "*" --out out/eval
python3 experiments/evaluate.py --world ac --conditions L M H HML shock --episodes "$EPISODES" --pattern "ac_*" --out out/eval_ac
python3 experiments/evaluate.py --conditions M shock --episodes 500 --pattern "lob_*_mixed_s*" --calib out/calib/regimes_fastres.json --out out/eval_sens_fastres
python3 experiments/evaluate.py --conditions M shock --episodes 500 --pattern "lob_*_mixed_s*" --latency 0.5 --out out/eval_sens_latency
python3 experiments/evaluate.py --conditions M shock --episodes 500 --pattern "lob_*_mixed_s*" --fee-mult 3 --out out/eval_sens_fees
echo "== 9. report"
mkdir -p out/report
python3 experiments/report.py --eval out/eval --eval-ac out/eval_ac --runs runs --out out/report | tee out/report/report.txt
python3 experiments/extra_analysis.py | tee out/report/extra.txt
echo "== 10. demonstration episode (Fig. 5.7)"
python3 experiments/demo.py
