#!/usr/bin/env bash
# v2 study (risk-averse DDQN): training grid over lam', evaluation, report. Phase-1 files are not touched.
# Writes runs_v2/ and out_v2/. PROCS parallel jobs (one per core).
set -euo pipefail
cd "$(dirname "$0")"
PROCS=${PROCS:-4}
mkdir -p runs_v2 out_v2
echo "== 1. tests"
python3 tests/test_sim.py
echo "== 2. training: DDQN at lam' in {0, 0.5, 1, 2, 4, 8} and PPO at 0.5, seeds 10-19"
python3 experiments/run_grid_lam.py --procs "$PROCS" | tee -a runs_v2/grid.log
echo "== 3. evaluation on settings A-D3 (1,000 episodes each), AC solved for each lam'"
python3 experiments/evaluate_lam.py --episodes 1000 --out out_v2/eval --procs "$PROCS" | tee out_v2/eval.log
echo "== 4. sensitivity (500 episodes, settings B and D2)"
python3 experiments/make_variant.py --kappa-mult 2 --out out/calib/regimes_fastres.json
python3 experiments/evaluate_lam.py --conditions M shock --episodes 500 --calib out/calib/regimes_fastres.json --out out_v2/eval_sens_fastres --procs "$PROCS" | tee out_v2/eval_sens.log
python3 experiments/evaluate_lam.py --conditions M shock --episodes 500 --latency 0.5 --out out_v2/eval_sens_latency --procs "$PROCS" | tee -a out_v2/eval_sens.log
python3 experiments/evaluate_lam.py --conditions M shock --episodes 500 --fee-mult 3 --out out_v2/eval_sens_fees --procs "$PROCS" | tee -a out_v2/eval_sens.log
echo "== 5. report"
python3 experiments/report_lam.py | tee out_v2/report.txt
