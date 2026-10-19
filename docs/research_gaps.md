# Research gaps in prior work, and how this implementation addresses them

Each row names a documented weakness, where it comes from, what the code does about it, and where the
evidence is. "Deck gap n" refers to slide 7 of the BTP deck.

| # | Gap or flaw | Source | Fix in this implementation | Evidence |
| --- | --- | --- | --- | --- |
| 1 | Linear impact is assumed, not measured | Almgren and Chriss (2000). Macrì and Lillo (2024) list linear impact as their own limitation. Deck gaps 5 and slide 4 | Impact is not a formula. It comes from orders consuming calibrated queues, with refill, quote revision and resilience. The exponent is then measured | Metaorder impact grows as X^0.59 to X^0.71 by regime; every 95% CI excludes linear (3,000 seeds). That is concave, within the empirical range (Almgren et al., 2005: about 0.6; Tóth et al., 2011: about 0.5). Temporary-impact exponent is 0.65 to 0.78, so AC's linear η is misspecified (`out/validation`, `out/calib/ac_params.json`) |
| 2 | Simulators use stylised or borrowed parameters | Macrì and Lillo (2024): no market data. arXiv 2511.15262 (QRM + DDQN): one calibration to one stock (France Télécom, 2010–2012), one fixed regime. Deck gap 2 | Per-regime maximum-likelihood queue-reactive rates from LOBSTER, then a method-of-simulated-moments stage. Moments are weighted by block-bootstrap standard errors, so the fit is judged against sampling error | `out/validation/fit_final.log` (final simulator, 200 / 1,000 fresh seeds per regime): spread, depth over 5 levels and level-1 queue within 14% in every regime; volatility too high by 14% in L, 34% in M and 37% in H. The 10-second market-order response (4.4 vs 5.1 ticks, t = -1.8) and VR(60) (t = -1.1) are within two standard errors. VR(300) (t = +2.1), sign autocorrelation at lags 1 to 10, and 2 to 5 minute reversion are not |
| 3 | Historical replay ignores how the market reacts to the agent | Cheridito and Weiss (2025). Historical-replay RL such as Ning, Lin and Jaimungal (2021) | Generative simulator. The agent's orders deplete queues, trigger post-trade quote revision (Hasbrouck, 1991) and move the latent fundamental by the permanent share | Within 5 minutes of completion, 17 to 19% of peak metaorder impact reverts (95% CIs 12 to 22%). That is partial reversion, below the roughly one third in Bershova and Rakhlin (2013); see the limitations. Counterfactual runs use common random numbers |
| 4 | Liquidity is static within an episode, or the agent observes it | Macrì and Lillo (2024): impact follows deterministic trends or a stochastic process; they argue liquidity is latent. Deck gap 3 | Regime switches within an episode: scripted H→M→L, a sudden H→L shock with 50% depth withdrawn, and Markov switching with the transition matrix estimated from the data. The agent sees only observable proxies (spread, depth, order-flow imbalance, volume, volatility) | Oracle ablation: agents also trained with the true regime label, to measure the value of latent-liquidity information |
| 5 | No controlled comparison of value-based and policy-gradient RL | Deck gaps 1 and 7. Tonkin et al. (2025): RL execution studies are hard to compare | Identical observation, 18 actions, reward, network body (2×128), environment-step budget, tuning budget (4 configurations each) and validation seeds. Both evaluated deterministically on the same episodes | `runs/tuning.csv`, `out/report/ddqn_vs_ppo.csv` |
| 6 | Few seeds and point estimates | Macrì and Lillo (2024) report test-set mean and standard deviation without training-seed variation. Henderson et al. (2018) and Agarwal et al. (2021) show this misleads | 5 seeds per method and condition. Stratified bootstrap intervals, interquartile mean, probability of improvement. Paired, common-random-number comparisons with baselines | `experiments/report.py` |
| 7 | RL compared only with weak baselines | TWAP-only comparisons are common, and Tonkin et al. (2025) call for benchmarking | Baselines: TWAP, passive TWAP, conditional VWAP (dynamic volume forecast, Białkowski et al., 2008), POV, AC calibrated in each regime, and an adaptive AC that re-solves its schedule with an online regime classifier | `liqexec/policies.py` |
| 8 | Restricted actions: market orders only, no queue position | Macrì and Lillo sell shares at the mid less linear impact. 2511.15262 uses market orders sized to the best ask. Cheridito and Weiss (2025) criticise restricted order placement | Market order, limit order joining the best ask (behind the queue), or limit order improving by one tick (front of the queue). Queue position falls with trades and cancellations, and fills can be partial | `liqexec/lobsim.py` (`_buy_mo`, cancellations at the agent's level) |
| 9 | Average shortfall hides tail risk | Deck gap 4 | Standard deviation, CVaR95, shock surge (shock minus High), and degradation ratio (Low over High) | `out/report/robustness.csv` |
| 10 | Frictions are left out | Deck gap 6 | Maker/taker fees (NASDAQ-like: $0.0030 taker, $0.0020 rebate), latency between decision and arrival, partial fills, forced completion at T | Sensitivity runs: latency 0.5 s and fees ×3 |
| 11 | Simulator misspecification is never tested | Common in RL execution work. Vyetrenko et al. (2020) on realism metrics | Untargeted realism checks. The remaining misfit (too little reversion of background impact over 2 to 5 minutes) is bracketed by a second simulator with twice the resilience; trained agents are re-evaluated there | `out/eval_sens_fastres` |
| 12 | The queue-reactive model assumes a large-tick stock | Huang, Lehalle and Rosenbaum (2015) index queues by tick distance from a reference price | Queues are indexed by occupied level, and gaps are drawn from the data (AAPL 2012: median spread 15 ticks) | `liqexec/calibrate.py` |
| 13 | The queue-reactive model has Poisson, sign-independent market orders | Data: lag-1 sign autocorrelation 0.63; 44% of market orders arrive within 10 ms of the previous one | Hawkes cluster representation (branching ratio 0.46, estimated by maximum likelihood). Long-memory signs from Pareto metaorders (Lillo, Mike and Farmer, 2005) | `out/calib/msm.log` |

## Corrections to the deck itself

* **Reward sign (slide 9).** `R_t = -(v_t P_t + λσ²q_t²)` treats a seller's revenue `v_t P_t` as a cost, so the agent would learn not
  to sell. The implementation uses shortfall against the arrival price: `r_t = Σ q (p - S_0)/(X S_0)·10⁴ - fees - λ (q_{t+1}/X)²`,
  whose sum is `-IS - risk penalty` (checked in `tests/test_sim.py`).
* **Undefined `I_t` in the state (slide 9).** If it is the liquidity regime, giving it to the agent contradicts slide 3, which says
  liquidity is latent. The state holds only observable features. The regime label is used in one ablation only.
* **One symbol, two meanings.** γ is the permanent-impact coefficient on slide 4 and the impact exponent on slides 3 and 11.
  The code uses γ for permanent impact and δ for the exponent.
* **Novelty claim on intra-episode shocks (slide 13).** Macrì and Lillo (2024) already let liquidity vary within an episode,
  through deterministic trends and a stochastic process. What is new here is shock evaluation in a data-calibrated
  limit order book with endogenous impact, and the head-to-head comparison of DDQN and PPO there.
* **Experiment labels.** The deck and the report use A = High, B = Medium, C = Low. The code printed A = Low
  until 14 Oct 2026; `experiments/report.py` now uses the deck's labels.
* **Citation details.** Macrì and Lillo is *Applied Mathematical Finance* 31(5), 2024 (arXiv:2402.12049); the deck says 2025.

## Limitations that remain

* The agents use an MLP over 13 hand-built features. The LSTM and Transformer encoders in the Review-1 plan are
  deferred, and no explicit impact kernel G(τ) is fitted (impact is emergent and its exponent is measured).
* Volatility in the simulator is 14 to 37% higher than in the data (`out/validation/fit_final.log`).

* The data is one stock on one day. Regimes are intraday 5-minute windows, so the contrast between High and Low is moderate
  (depth over 5 levels: 983 vs 604 shares; spread: 12.7 vs 17.3 ticks). The pipeline accepts more LOBSTER days.
* Hidden liquidity (LOBSTER type 5 executions, about 35% of executed volume) is left out, which is conservative for
  market orders.
* In the simulator, impact reverts less over 2 to 5 minutes than in the data: background flow (response ratio) and the agent's
  metaorders (17 to 19% vs about one third). A second simulator with twice the resilience brackets this, and every headline
  conclusion holds in both.
* Metaorder impact is checked against published ranges, not against this dataset, because LOBSTER has no metaorder labels.

## References

Agarwal, R. et al. (2021). Deep RL at the edge of the statistical precipice. NeurIPS.
Almgren, R. and Chriss, N. (2000). Optimal execution of portfolio transactions. Journal of Risk.
Almgren, R., Thum, C., Hauptmann, E. and Li, H. (2005). Direct estimation of equity market impact. Risk.
Bacry, E. et al. (2015). Market impacts and the life cycle of investors orders. Market Microstructure and Liquidity.
Bershova, N. and Rakhlin, D. (2013). The non-linear market impact of large trades. Quantitative Finance.
Białkowski, J., Darolles, S. and Le Fol, G. (2008). Improving VWAP strategies: a dynamic volume approach. Journal of Banking & Finance.
Bouchaud, J.-P. et al. (2004). Fluctuations and response in financial markets. Quantitative Finance.
Byrd, D., Hybinette, M. and Balch, T. (2020). ABIDES: towards high-fidelity multi-agent market simulation. ACM SIGSIM-PADS.
Cheridito, P. and Weiss, M. (2025). Reinforcement learning for trade execution with market and limit orders. arXiv:2507.06345.
Hasbrouck, J. (1991). Measuring the information content of stock trades. Journal of Finance.
Henderson, P. et al. (2018). Deep reinforcement learning that matters. AAAI.
Huang, W., Lehalle, C.-A. and Rosenbaum, M. (2015). Simulating and analyzing order book data: the queue-reactive model. JASA.
Lillo, F., Mike, S. and Farmer, J. D. (2005). Theory for long memory in supply and demand. Physical Review E.
Macrì, A. and Lillo, F. (2024). Reinforcement learning for optimal execution when liquidity is time-varying. Applied Mathematical Finance.
Ning, B., Lin, F. H. T. and Jaimungal, S. (2021). Double deep Q-learning for optimal execution. Applied Mathematical Finance.
Perold, A. (1988). The implementation shortfall: paper versus reality. Journal of Portfolio Management.
Reinforcement learning in queue-reactive models: application to optimal execution (2025). arXiv:2511.15262.
Tonkin, I., Gepp, A., Harris, G. and Vanstone, B. (2025). Benchmarking deep reinforcement learning approaches to trade execution. Pacific-Basin Finance Journal.
Tóth, B. et al. (2011). Anomalous price impact and the critical nature of liquidity in financial markets. Physical Review X.
Vyetrenko, S. et al. (2020). Get real: realism metrics for robust limit order book market simulations. ICAIF.
