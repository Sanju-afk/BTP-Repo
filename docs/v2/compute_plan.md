# CVaR distributional RL extension: compute estimate and schedule

Nothing below has been trained yet. Training starts only after SANJU's go-ahead.

## What was measured (cloud container, 4 cores, torch 2.14.1, numba)

| Measurement | Value | Source |
|---|---|---|
| `tests/test_sim.py` (4 tests) | pass, 9.8 s | run here |
| DDQN, 25k steps, mixed, tuned hp (lr 1e-4, batch 128, train_freq 4) | 79 s, 1 core | run here (throwaway seed 98, scratch dir) |
| PPO, 25k steps, mixed, tuned hp | 35 s, 1 core | run here (throwaway seed 99, scratch dir) |
| One gradient update, batch 128: scalar Q head / 32 quantiles / 50 quantiles | 2.8 / 7.1 / 11.6 ms | micro-benchmark here |
| DDQN 250k (Phase-1 log) | 318–493 s, mean 434 s | `runs_grid.log` |
| PPO 250k (Phase-1 log) | 205–397 s, mean 339 s | `runs_grid.log` |

The Phase-1 logs were produced in an environment we have not seen, so I use the slower numbers measured
here as the planning figure. The short runs here scale to roughly DDQN 14 min and PPO 6 min per 250k-step
run on one core. QR-DDQN does the same environment steps and the same number of updates (62.5k at
train_freq 4); each update costs about 4–6 ms more with K = 40 quantiles, so about 19 min per run.

## Runs and core-hours (planning figure, one run = one core)

| Block | Runs | Each | Core-hours |
|---|---|---|---|
| Tuning: QR-DDQN risk-neutral and QR-DDQN-CVaR(0.25), same 4-config grid as DDQN (lr {1e-4, 5e-4} × {(64,1), (128,4)}), 150k steps, 150 val episodes × 6 conditions | 8 | 11–25 min (train_freq 1 configs take ~2×) | ≈ 2.4 |
| Main grid, mixed, 250k steps, 10 fresh seeds: DDQN, PPO, QR-DDQN-RN, QR-DDQN-CVaR(0.25) | 40 | 6–19 min | ≈ 9.7 |
| Secondary arm QR-DDQN-CVaR(0.1), 10 seeds (uses the 0.25 arm's tuned hp) | 10 | 19 min | ≈ 3.2 |
| Optional ablation: threshold-augmented CVaR (Lim & Malik, 2022), 10 seeds | 10 | 19 min | ≈ 3.2 |
| Evaluation: 1,000 test episodes × 6 settings (A–D3) per model, plus AC-world and 3 sensitivity variants | 50–60 models | — | ≈ 1.5 |
| **Primary set (tuning + 4 arms + α = 0.1 + eval)** | | | **≈ 17** |
| **With the threshold ablation** | | | **≈ 20** |

Wall-clock: about 4.5–5 h on 4 cores, or 8–10 h on 2 cores (the team laptop setting in Phase 1). If the
Phase-1 log timings hold on your machine, halve these.

## Proposed schedule

1. **Code and tests (no training).** `liqexec/qr_ddqn.py` (new agent, same MLP body, K = 40 quantiles so
   αK is whole for α = 0.1 and 0.25), registered without touching DDQN/PPO; `experiments/train_v2.py`,
   `run_grid_v2.py`, `tune_v2.py` writing to `runs_v2/`; `evaluate.py` reused with `--out out_v2/...`.
   Unit tests: quantile loss sign and shape, CVaR selection on a toy example, reward-sum identity.
   `reproduce.sh` re-checked unchanged.
2. **Pre-registration file** committed before any run (`docs/preregistration_v2.md`): arms, seeds 10–19,
   primary contrasts (CVaR95 of IS vs DDQN averaged over A–D3; mean-IS non-inferiority margin +0.3 bps),
   smallest effect of interest on CVaR95, Holm correction for secondary results.
3. **Tuning** (≈ 2.4 core-hours) → `runs_v2/tuned.json`.
4. **Main grid** (≈ 13 core-hours with α = 0.1) → `runs_v2/`.
5. **Evaluation and report** → `out_v2/`: results table, cost histograms, CVaR-vs-mean scatter with CIs,
   % sold before T, forced-sweep share, quantile-spread check against Monte Carlo IS.
6. **Slides and report section.**

All of it can run in this cloud container; trained models get committed to the branch as each block
finishes, since the container is not permanent.
