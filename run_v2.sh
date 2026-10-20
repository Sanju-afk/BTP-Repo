#!/usr/bin/env bash
# v2 study (CVaR distributional RL): tuning, training grid, evaluation. Phase-1 files are not touched.
# Writes runs_v2/ and out_v2/. PROCS parallel jobs (one per core).
set -euo pipefail
cd "$(dirname "$0")"
PROCS=${PROCS:-4}
mkdir -p runs_v2 out_v2
echo "== 1. tests"
python3 tests/test_sim.py && python3 tests/test_qr.py
echo "== 2. equal-budget tuning of QRDDQN-RN and QRDDQN-CVaR25"
[ -f runs_v2/tuned.json ] || python3 experiments/tune_v2.py --steps 150000 --val-episodes 150 --procs "$PROCS" | tee runs_v2/tune.log
echo "== 3. training grid: 6 arms x seeds 10-19"
python3 experiments/run_grid_v2.py --procs "$PROCS" | tee -a runs_v2/grid.log
echo "== 4. evaluation on settings A-D3 (1,000 episodes each) and sensitivity variants (500 episodes)"
python3 experiments/evaluate_v2.py --episodes 1000 --out out_v2/eval --procs "$PROCS" | tee out_v2/eval.log
python3 experiments/make_variant.py --kappa-mult 2 --out out/calib/regimes_fastres.json
python3 experiments/evaluate_v2.py --conditions M shock --episodes 500 --no-baselines --calib out/calib/regimes_fastres.json --out out_v2/eval_sens_fastres --procs "$PROCS" | tee out_v2/eval_sens.log
python3 experiments/evaluate_v2.py --conditions M shock --episodes 500 --no-baselines --latency 0.5 --out out_v2/eval_sens_latency --procs "$PROCS" | tee -a out_v2/eval_sens.log
python3 experiments/evaluate_v2.py --conditions M shock --episodes 500 --no-baselines --fee-mult 3 --out out_v2/eval_sens_fees --procs "$PROCS" | tee -a out_v2/eval_sens.log
echo "== 5. report"
python3 experiments/report_v2.py | tee out_v2/report.txt
