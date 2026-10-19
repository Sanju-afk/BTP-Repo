# Pre-registration: tail-risk-aware distributional RL for execution (v2)

Committed before any v2 tuning or training run. Anything not listed here is exploratory and labelled so.

## Question

Does a QR-DDQN agent that selects actions on the lower tail of its learned return distribution reduce the
tail of the implementation shortfall relative to DDQN, without giving up more than 0.3 bps of mean
shortfall? And is the DDQN/PPO tail gap of Phase 1 explained by risk-sensitivity rather than by learning a
distribution as such?

## Fixed setting (unchanged from Phase 1)

Simulator `out/calib/regimes.json`, environment `liqexec/env.py` (13 features, 18 actions, reward with
λ′ = 0.5 bps, forced sweep at T), MLP 13→128→128, test episodes 10,000,000 + i (i < 1,000) per setting,
settings A = H, B = M, C = L, D1 = HML, D2 = shock, D3 = markov, all agents trained on "mixed" for 250k steps.

## Arms

| Arm | Agent | Hyperparameters |
|---|---|---|
| DDQN | Phase-1 DDQN, re-run | `runs/tuned.json` |
| PPO | Phase-1 PPO, re-run | `runs/tuned.json` |
| QRDDQN-RN | QR-DDQN, risk-neutral (mean of quantiles) | tuned, 4 configs |
| QRDDQN-CVaR25 (**primary**) | QR-DDQN, CVaR₀.₂₅-greedy | tuned, 4 configs, α fixed |
| QRDDQN-CVaR10 (secondary) | QR-DDQN, CVaR₀.₁-greedy | those of CVaR25 |
| QRDDQN-THR25 (ablation) | QR-DDQN, threshold rule for static CVaR₀.₂₅ (Lim & Malik, 2022) | those of CVaR25 |

QR-DDQN: K = 40 quantiles at τᵢ = (2i−1)/80, quantile Huber loss with κ = 1, Double-DQN target with the
arm's own criterion; all other settings as DDQN. Tuning grid and protocol as Phase-1 DDQN
(`experiments/tune_v2.py`). Training seeds **10–19** for every arm (fresh; Phase-1 used 0–4).

## Primary contrasts (QRDDQN-CVaR25 minus DDQN)

Paired by training seed and by test episode. The 95% interval comes from a two-level bootstrap (2,000
resamples): resample seeds, then episodes within each seed, applied jointly to both arms.

1. **Tail.** CVaR₉₅ of IS (mean of the worst 5% of episodes, per run), averaged over the six settings.
   Smallest effect of interest: **1.0 bps**. Reported as: reduction if the interval lies below 0; a
   meaningful reduction if it also lies below −1.0; otherwise "no tail benefit beyond the interval's bound".
2. **Mean.** Mean IS averaged over the six settings. Non-inferior if the upper bound of the interval is
   below **+0.3 bps**.

## Secondary results (Holm-corrected across the family, or labelled exploratory)

Per-setting versions of 1 and 2; objective; every QR arm vs DDQN and vs PPO; QRDDQN-RN vs DDQN (effect of
the distribution alone); CVaR25 vs RN (effect of the risk criterion); α sensitivity (CVaR10); the threshold
ablation; limit-order share; share sold before T; inventory at the D2 shock; learning curves.

## Mechanism check

Forced-sweep share (1 − sold before T) and its link to tail cost: share of the worst-5% episodes' cost from
the forced order, per arm; cost histograms; learned quantile spread at t = 0 compared with the Monte Carlo
IS distribution of the same policy on the test episodes (calibration of the learned distribution).

## Rules

New code only in new files (`liqexec/qr_ddqn.py`, `experiments/*_v2.py`, `tests/test_qr.py`); outputs only
in `runs_v2/` and `out_v2/`. Every reported number comes from generated files. Front-loading is treated as
the correct risk-averse behaviour (Almgren & Chriss, 2000), not a defect.
