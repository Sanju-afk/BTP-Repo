#!/usr/bin/env bash
# Regenerate every reported number from the saved models, without recalibrating or retraining
# (run_all.sh steps 8-10). About 35 minutes on 2 cores. Compare the result with the shipped report:
#   ./reproduce.sh && diff out/report/report.txt out/report_shipped/report.txt
set -euo pipefail
cd "$(dirname "$0")"
EPISODES=${EPISODES:-1000}
[ -d out/report_shipped ] || cp -r out/report out/report_shipped
python3 tests/test_sim.py
python3 experiments/evaluate.py --episodes "$EPISODES" --pattern "*" --out out/eval
python3 experiments/evaluate.py --world ac --conditions L M H HML shock --episodes "$EPISODES" --pattern "ac_*" --out out/eval_ac
python3 experiments/make_variant.py --kappa-mult 2 --out out/calib/regimes_fastres.json
python3 experiments/evaluate.py --conditions M shock --episodes 500 --pattern "lob_*_mixed_s*" --calib out/calib/regimes_fastres.json --out out/eval_sens_fastres
python3 experiments/evaluate.py --conditions M shock --episodes 500 --pattern "lob_*_mixed_s*" --latency 0.5 --out out/eval_sens_latency
python3 experiments/evaluate.py --conditions M shock --episodes 500 --pattern "lob_*_mixed_s*" --fee-mult 3 --out out/eval_sens_fees
python3 experiments/report.py --eval out/eval --eval-ac out/eval_ac --runs runs --out out/report | tee out/report/report.txt
python3 experiments/extra_analysis.py | tee out/report/extra.txt
python3 experiments/demo.py | tee out/report/demo.txt
