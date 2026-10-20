# Pre-registration: risk-averse DDQN (v2)

Committed before any v2 training run. Anything not listed here is exploratory and labelled so.

## Question

Phase 1 found that DDQN has the lower mean shortfall and PPO the lower tail (CVaR95), and traced DDQN's tail to
the inventory it leaves for the forced market order at T. With the agent unchanged (plain DDQN), does raising
the risk penalty λ′ in the reward move DDQN along a mean–tail trade-off? In particular, is there a λ′ that
lowers CVaR95 by at least 1 bps while raising mean shortfall by less than 0.3 bps? And how does the DDQN
trade-off compare with the Almgren–Chriss schedule solved for the same λ′?

## Fixed setting (unchanged from Phase 1)

Simulator `out/calib/regimes.json`, environment `liqexec/env.py`, DDQN and PPO in `liqexec/agents.py` with the
Phase-1 tuned hyperparameters (`runs/tuned.json`; not re-tuned per λ′), 250k steps on "mixed", test episodes
10,000,000 + i (i < 1,000), settings A = H, B = M, C = L, D1 = HML, D2 = shock, D3 = markov.

## Arms

| Arm | What | Seeds |
|---|---|---|
| DDQN@λ′, λ′ ∈ {0, 0.5, 1, 2, 4, 8} bps | DDQN trained with reward penalty λ′(q/X)² per interval | 10–19 |
| PPO@0.5 | Phase-1 PPO, reference | 10–19 |
| AC@λ′, same λ′ values | Almgren–Chriss and AC-adaptive solved for λ′ (`lam_from_penalty`) | deterministic |
| TWAP, TWAP-LO, VWAP, POV | Phase-1 baselines (do not depend on λ′) | deterministic |

DDQN@0.5 is the Phase-1 configuration re-run on the fresh seeds and is the reference arm.

## Contrasts

Paired by training seed and by test episode, 95% intervals by two-level bootstrap (seeds, then episodes; 2,000
resamples), averaged over the six settings. Shortfall (IS) excludes the penalty, so it is comparable across λ′.

1. **Tail.** CVaR₉₅ of IS, DDQN@λ′ minus DDQN@0.5, for λ′ ∈ {1, 2, 4, 8}. Smallest effect of interest 1.0 bps.
2. **Mean.** Mean IS, same pairs. Non-inferiority margin +0.3 bps.
3. **Headline.** A λ′ "keeps the mean and cuts the tail" if the tail interval lies below −1.0 and the mean
   interval's upper bound is below +0.3. Holm correction over the four λ′ values for tests 1 and 2.
4. **Against Almgren–Chriss.** DDQN@λ′ minus AC@λ′ on the objective (IS + λ′ penalty), the quantity both
   optimise, for every λ′; Holm-corrected over the six values.

## Secondary results (labelled exploratory)

Per-setting versions; DDQN@λ′ vs PPO@0.5; share sold by the forced order at T; limit-order share; inventory at
the D2 shock; learning curves; DDQN@0 vs DDQN@0.5.

## Rules

New files only (`experiments/*_lam.py`, `run_v2.sh`); outputs only in `runs_v2/` and `out_v2/`. Every reported
number comes from generated files. Front-loading is treated as the correct risk-averse behaviour (Almgren &
Chriss, 2000), not a defect.
