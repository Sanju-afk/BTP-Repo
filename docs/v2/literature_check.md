# CVaR-greedy distributional agents fill a real execution gap

**Bottom line.** A QR-DDQN agent that picks actions by CVaR_α (α ∈ {0.1, 0.25}) is a defensible extension for the BTP, and on the evidence gathered it is novel as a *combination*, not as a new technique.

**What the search found.**
- No verifiable paper uses distributional RL (C51, QR-DQN, IQN, FQF) or CVaR action selection for optimal execution. Every deep-RL execution paper opened during the research optimises a risk-neutral expected shortfall or revenue.
- CVaR execution has been solved by dynamic programming, but not with RL. Min, Moallemi & Maglaras do this in an Almgren–Chriss-type model.
- Risk-measure RL exists in finance, but not for execution. Coache & Jaimungal's work covers hedging, statistical arbitrage and portfolios.
- The algorithm comparison on its own is not new. **Schnaubelt (2022) already compared DDQN and PPO for execution** and found PPO best. **Espana et al. (2025) already ran DDQN in a calibrated queue-reactive LOB.**

**Where the novelty sits.** It rests on four things together:
- a CVaR-sensitive distributional agent;
- a controlled three-way comparison;
- intra-episode liquidity shocks;
- an explicit link from tail shortfall to the forced terminal trade. No paper found quantifies that link.

**Caveats the team must state.**
- **Greedy per-step CVaR selection does not optimise the static CVaR of total cost** (Lim & Malik 2022). The team's draft wording "CVaR-optimising" must change.
- An accumulated-shortfall input feature cannot fix this, because CVaR is translation-equivariant.
- A threshold-augmented rule (Bäuerle & Ott 2011; Lim & Malik 2022) does target static CVaR. It costs roughly 10–20 lines of code and some sample efficiency.

**Backup idea.** Adversarial training over liquidity-shock timing and size for *execution* also appears open. The nearest prior art is in market making.

**Citations to fix.** Several citations need checking before the panel, and "Koh et al. 2023" could not be found at all.

**How citations are tagged.** Every citation below carries one of three tags:
- **[V]**: the page was opened, or the paper was seen on a primary or indexing site, during the research.
- **[M]**: the detail comes from the researchers' memory and was not re-checked.
- **[U]**: unverified, or searched for and not found.

## Execution RL is almost entirely risk-neutral, so the CVaR slot is empty

**The core gap.** The search covered value-based, policy-gradient and risk-measure RL for liquidating a fixed inventory within a horizon. It found no execution paper that trains on a CVaR or distributional objective.

**Value-based execution papers, all risk-neutral.**
- Tabular Q-learning started the line. Nevmyvaka, Feng & Kearns (ICML 2006) **[V]** is the origin; its contents were not opened, and its LOB-replay design is recalled **[M]** ([mlanthology](https://mlanthology.org/icml/2006/nevmyvaka2006icml-reinforcement)).
- Ning, Lin & Jaimungal **[V]** brought in Double DQN. They report mean and median outperformance against TWAP, and the abstract does not describe a risk term ([arXiv:1812.06600](https://arxiv.org/abs/1812.06600v1)).
- Macrì & Lillo **[V]** apply DDQN to an Almgren–Chriss model with **time-varying temporary and permanent impact**. This is the nearest "liquidity regime" execution work. It has no LOB, no discrete shocks and no tail metrics ([arXiv:2402.12049](https://arxiv.org/abs/2402.12049)).
- Hafsi & Vittori **[V]** use DQN in ABIDES with a **risk-neutral expected-IS objective**. They report IS variance only, and penalise each unexecuted share by 5 rather than forcing liquidation ([arXiv:2411.06389](https://arxiv.org/pdf/2411.06389v2)).

**Policy-gradient execution papers, also risk-neutral.**
- PPO: Lin & Beling (IJCAI 2020) ([IJCAI](https://www.ijcai.org/Proceedings/2020/627)) **[V]**; the industrial PPO-LSTM system of Byun et al. ([IDEAS](https://ideas.repec.org/a/gam/jfinte/v2y2023i3p23-429d1182401.html)) **[V]**; PPO with MAP-Elites liquidity×volatility regime specialists in de Witt & Pakkanen ([arXiv:2601.22113](https://arxiv.org/html/2601.22113v1)) **[V]**.
- Cheridito & Weiss use a logistic-normal policy gradient. It **maximises expected revenue** ([arXiv:2507.06345](https://arxiv.org/abs/2507.06345)) **[V]**.
- Borjigin & He come closest to tail awareness, but their PPO agent **reports CVaR95 only and does not optimise it**. They explicitly leave risk-constrained RL to future work ([arXiv:2510.04952](https://arxiv.org/html/2510.04952v2)) **[V]**.

**CVaR execution exists outside RL.** Min, Moallemi & Maglaras solve a scaled CVaR execution objective by DP/HJB **[V]** ([PDF](https://moallemi.com/ciamac/papers/cvar-exec-2022.pdf)):
- the model is continuous-time, with temporary quadratic impact;
- they augment the state with a quantile process, which turns the problem into a game against an adversary;
- the optimal policy is **liquidate-only and "aggressive-in-the-money"**;
- it beats the optimal static schedule by about 5–15% and an optimised VWAP by about 15–25%.

That policy shape gives the team a testable hypothesis: a CVaR agent should front-load selling to reduce its exposure at the terminal liquidation.

**Risk-measure RL exists, but not for execution.**
- Coache & Jaimungal (*Mathematical Finance* 2024) **[V]** give time-consistent dynamic convex risk measures with an actor-critic. Their examples are statistical arbitrage, hedging and obstacle avoidance ([arXiv:2112.13414](https://arxiv.org/abs/2112.13414v1)).
- Coache, Jaimungal & Cartea **[V]** cover statistical arbitrage and portfolio allocation ([arXiv:2206.14666](https://arxiv.org/abs/2206.14666?context=cs)).
- Jaimungal, Pesenti, Wang & Tatsat (*SIAM JFM* 2022) **[V]** cover robust portfolio allocation, benchmark optimisation and statistical arbitrage ([arXiv:2108.10403](https://arxiv.org/abs/2108.10403)).

**How the adjacent finance problems handle tail risk.**
- Deep hedging minimises a convex risk measure of terminal P&L by direct policy optimisation, so it never faces the per-step heuristic problem ([IDEAS](https://ideas.repec.org/a/taf/quantf/v19y2019i8p1271-1291.html)) **[V]**; the method summary is **[M]**.
- Cao et al. optimise mean + c·std of hedging cost with two critics ([arXiv:2103.16409](https://arxiv.org/abs/2103.16409v1)) **[V]**.
- Market-making RL handles risk through reward shaping (Spooner et al. 2018) **[V]** or an adversary (Spooner & Savani 2020) **[V]**.

**Searches that came back empty.** Targeted queries for "distributional reinforcement learning optimal execution", "CVaR optimal execution reinforcement learning" and quantile-DQN liquidation turned up only other domains. Examples are multi-agent RL (RiskQ), portfolio management and gas futures ([RiskQ](https://arxiv.org/html/2311.01753)).

**Gaps in the search.** It did not cover Google Scholar or SSRN full text, ICAIF 2023–2025 tables of contents, or thesis repositories. An MSc or PhD thesis on QR-DQN execution could exist.

**A referee risk.** Prasad's 2026 audit of distributional-RL risk claims **[V]** refutes **40–95% of the strongest claimed risk trade-offs on MinAtar**. Acting on CVaR heads ranged "from helpful to worse than chance" ([arXiv:2607.11607](https://arxiv.org/html/2607.11607)). The audit does not cover execution, but a referee could raise it. The study should therefore validate the learned quantiles against Monte Carlo IS distributions from the simulator.

## Eight near-neighbours each miss at least two of five components

The proposed study has five components:
- **(a)** DDQN vs PPO vs a distributional CVaR agent;
- **(b)** a CVaR objective;
- **(c)** a data-calibrated reactive LOB;
- **(d)** liquidity shocks;
- **(e)** tail shortfall attributed to forced terminal liquidation.

No paper found covers more than two or three of them.

| Closest work | What overlaps | How the BTP differs |
|---|---|---|
| Espana, Hafsi, Lillo & Vittori 2025 **[V]** ([arXiv:2511.15262](https://arxiv.org/html/2511.15262)) | (c): DDQN in a Huang–Lehalle–Rosenbaum queue-reactive model calibrated on France Télécom data. Leftover inventory is liquidated in an extra step, and that cost is counted in IS | DDQN only, risk-neutral, no shocks (only θ/θ^reinit sweeps). Reports means and standard deviations only (DDQN −0.259±0.631 vs TWAP −0.365±0.652) |
| Schnaubelt 2022, *EJOR* **[V]** ([IDEAS](https://ideas.repec.org/a/eee/ejores/v296y2022i3p993-1006.html); [WP](https://www.iwf.rw.fau.de/files/2020/04/05_2020.pdf)) | (a) partly: DDQN vs PPO. Leftover volume is sent as a market order at the last step | Replayed crypto LOB (not reactive), no shocks, no CVaR. PPO cuts shortfall about 8.53% vs about 3.61% for DDQN, and ranks first in 89.06% of cases. Only mean and seed spread are reported |
| Hafsi & Vittori 2024 **[V]** ([arXiv:2411.06389](https://arxiv.org/html/2411.06389v2)) | Reactive ABIDES simulator. TRPO/PPO/A2C were tried, and DQN was "most reliable" | No numbers for the alternatives, terminal penalty instead of forced liquidation, no tail metrics |
| Macrì & Lillo 2024 **[V]** ([arXiv:2402.12049](https://arxiv.org/abs/2402.12049)) | (d) partly: time-varying impact | Almgren–Chriss model, not an LOB; no policy-gradient agent; no CVaR |
| Min, Moallemi & Maglaras 2022 **[V]** ([PDF](https://moallemi.com/ciamac/papers/cvar-exec-2022.pdf)) | (b): CVaR execution objective | Not RL; stylised model |
| Coache & Jaimungal 2024; Coache, Jaimungal & Cartea **[V]** | (b): RL that optimises dynamic CVaR or spectral risk | No execution application |
| Borjigin & He 2025 **[V]** ([arXiv:2510.04952](https://arxiv.org/html/2510.04952v2)) | Execution CVaR95 reported in ABIDES | PPO only; CVaR not optimised |
| de Witt & Pakkanen 2026 **[V]** ([arXiv:2601.22113](https://arxiv.org/html/2601.22113v1)) | Liquidity×volatility regimes | PPO only; propagator simulator; no CVaR |

**What this means for framing.**
- **Espana et al. is the direct baseline and should be cited prominently.** It uses the same simulator family, the same DDQN choice and the same convention of liquidating leftovers in an extra step.
- **Schnaubelt is the direct predecessor for the algorithm comparison.**
- The two disagree: Schnaubelt finds PPO ahead on replayed data, while Hafsi & Vittori report anecdotally that DQN is more reliable in ABIDES. The controlled comparison can be framed as resolving that disagreement in a reactive simulator.
- **No paper quantifies how much of the IS tail comes from the terminal forced trade**, even though every one of them uses a forced trade or a penalty. That attribution analysis is the least contested part of the contribution.
- Hendricks & Wilcox **[V]** provide indirect early evidence that risk-neutral RL widens IS dispersion. Their reward excluded variance, their leftover inventory went out as a market order, and the RL policy's IS standard deviation was **about 0.22% vs 0.14%** for Almgren–Chriss ([arXiv:1403.2229](https://arxiv.org/pdf/1403.2229)).

**Correcting the novelty sentence.** The draft reads: "to our knowledge, the first controlled comparison of value-based, policy-gradient and CVaR-optimising distributional agents for optimal execution in a data-calibrated, reactive LOB with liquidity shocks, linking tail-risk differences to forced end-of-horizon liquidation."

"To our knowledge" and the stacked qualifiers are the right instinct. Three edits are needed:
1. **"CVaR-optimising" is technically false** for CVaR-greedy selection (see the next section). Replace it with "CVaR-sensitive" or "CVaR-greedy".
2. "Reactive LOB" should name the model class ("queue-reactive"), so the claim is checkable against Espana et al.
3. The sentence should credit the predecessors in the same breath, so a panel member who knows Schnaubelt does not read it as an overclaim.

Suggested wording:

> "To our knowledge, this is the first study to compare value-based (DDQN), policy-gradient (PPO) and CVaR-sensitive distributional (QR-DDQN with CVaR_α-greedy action selection) agents for optimal execution under a common protocol, in a data-calibrated queue-reactive LOB simulator with intra-episode liquidity shocks, and to attribute differences in tail implementation shortfall to forced end-of-horizon liquidation. Prior work has compared DDQN and PPO on replayed order-book data (Schnaubelt 2022) and trained DDQN in a queue-reactive model without shocks or tail analysis (Espana et al. 2025); CVaR-optimal execution has been studied analytically (Min, Moallemi & Maglaras 2022) but not with RL."

**Two guards on that wording.**
- "Common protocol" must be literally true: same simulator, state features, seeds, step budget and evaluation episodes.
- The tail-attribution clause must be backed by an actual decomposition, for example the share of CVaR_α of IS that comes from the terminal trade, by agent.

## Greedy CVaR action selection optimises neither static nor nested CVaR

**What the agent does.** The planned rule picks a_t = argmax_a CVaR_α[Z(x_t, a)] at every step. Z is the return-to-go distribution, and CVaR is the mean of the lowest αK quantiles. This is the "Markov action-selection" scheme used by IQN's CVaR policies and by Keramati et al.

**What Lim & Malik prove.** Their Proposition 1 shows this scheme **"converges to neither the dynamic, Markovian CVaR nor the static CVaR,"** with counterexamples at γ = 1 ([NeurIPS 2022 PDF](https://proceedings.nips.cc/paper_files/paper/2022/file/c88a2bd0e793550d0e885aa6e31ca277-Paper-Conference.pdf)) **[V]**. Their main objection is not time-inconsistency as such. It is that the scheme does not converge to *either* well-defined target.

IQN itself never claimed full-return CVaR optimality. It notes that "it remains an open question as to exactly why we see improved performance for risk-averse policies" ([arXiv:1806.06923](https://arxiv.org/pdf/1806.06923)) **[V]**.

**Why the accumulated-shortfall feature does not fix it.** This is a derivation from the cited results, not a quoted claim.
- At step t the shortfall so far, c_t, is known.
- CVaR is translation-equivariant, so CVaR_α(c_t + G_t | F_t) = c_t + CVaR_α(G_t | F_t), where G_t is the cost-to-go.
- Adding c_t shifts every action's score by the same constant, so the argmax does not change.
- The policy therefore still applies a fresh α-tail at every step. The static objective instead applies the worst-α tail of *total* cost fixed at t = 0. That tail corresponds to different, path-dependent conditional tail levels later in the episode.
- The feature can still help the network represent the cost-to-go distribution. It does not change what is being optimised.

**The fix: carry a threshold.** Lim & Malik follow Bäuerle & Ott's state augmentation ([IDEAS](https://ideas.repec.org/a/spr/mathme/v74y2011i3p361-379.html)) **[V]** (theorem number not checked). The procedure:
1. Carry a threshold s alongside the state. Initialise it at the α-quantile of the initial return distribution, which is roughly θ_⌈αK⌉ of the chosen action.
2. Update it as s_{t+1} = (s_t − r_t)/γ. With γ = 1 this is simply s_t − r_t.
3. Act by argmax_a (1/K) Σ_i −(s_t − θ_i(x_t, a))^+.
4. Use the same W-based argmax for the Double-DQN target action: the online net selects and the target net evaluates. This is their Algorithm 2, which used N = 100 quantiles.

**Is that augmentation still sound?** Yes. Hau, Delage, Ghavamzadeh & Petrik (NeurIPS 2023; four authors, not three) **[V]** show that the popular *risk-level* augmentation of Chow et al. 2015 is **"inherently suboptimal regardless of the discretization level"**. They also state explicitly that the **accumulated-reward augmentation of Bäuerle & Ott "is unaffected"** ([arXiv v4](https://arxiv.org/html/2304.12477v4)).

**What the fix costs.**
- Code: about 10–20 extra lines.
- Sample efficiency: each transition may need to be seen under many thresholds.
- Theory: per Lim & Malik, "the convergence properties of the proposed operator remain unknown in the general settings".

**The exact alternative.** Bäuerle–Ott applied directly:
1. Train a risk-neutral agent on the hinge reward −(s − G)^+ for a grid of thresholds s, or train one threshold-conditioned network.
2. Pick the s that maximises s − (1/α)E[(s − G)^+].

With 20 steps this is feasible, but it multiplies the training runs.

**Recommendation.** Keep CVaR-greedy as the main agent, labelled honestly as a per-step risk-averse heuristic. Report the empirical CVaR of realised total IS on held-out episodes. Add the threshold rule as an ablation if time allows. A defensible methods sentence:

> "We use CVaR_α-greedy action selection over the learned quantile distribution of the return-to-go (Dabney et al. 2018; Keramati et al. 2020); this is a per-step risk-averse heuristic and is not guaranteed to optimise the static CVaR of total execution cost, nor a nested CVaR (Lim & Malik 2022, Prop. 1). We therefore evaluate CVaR of realised total shortfall empirically and compare against the threshold-augmented rule of Lim & Malik (2022), following Bäuerle & Ott (2011)."

Avoid "CVaR-optimal" and "time-consistent" anywhere in the thesis.

## Quantile count, loss reduction and reward scale are where QR-DDQN breaks

**Reference settings from the QR-DQN paper** ([arXiv:1710.10044](https://arxiv.org/pdf/1710.10044)) **[V]**:
- target fractions are midpoints τ̂_i = (2i − 1)/(2N);
- quantile Huber loss ρ^κ_τ(u) = |τ − δ{u<0}| L_κ(u), with **κ = 1**;
- **sum over predicted quantiles i and mean over target quantiles j**;
- tuned Atari values: **N = 200, Adam lr = 5e-5, Adam ε = 0.01/32**, with N searched over {10, 50, 100, 200};
- a target network as in DQN.

**Reference settings from IQN** ([arXiv:1806.06923](https://arxiv.org/pdf/1806.06923)) **[V]**:
- N = N′ = 8 samples were enough, with K = 32 policy samples;
- CVaR(η) is implemented by sampling τ ~ U[0, η], tested at η = 0.1;
- CVaR(0.1) helped strongly on Asterix and Assault but "suffers some loss" on Qbert and Space Invaders. Risk aversion is not a free lunch.

**Pitfalls for this setting.** These are implementation reasoning from the notes, not published guidance specific to execution.

- **Choose K so that αK is an integer for both α values.**
  - K = 40 gives 4 and 10 tail quantiles.
  - K = 100 gives 10 and 25.
  - K = 20 leaves only 2 quantiles at α = 0.1, which makes the action value noisy.
  - Report K and α together, and state the rounding rule if αK is fractional.
- **Use the paper's loss reduction.** Averaging over both i and j instead silently rescales the gradient by 1/K and changes the effective learning rate. Broadcast u_ij = target_j − pred_i so the |τ̂_i − 1{u<0}| weight attaches to the *predicted* quantile.
- **Scale the reward to κ.** κ = 1 assumes order-one rewards. Normalise shortfall in bps or currency, or scale κ to the reward units. No verified published guidance exists for non-Atari reward scales.
- **Keep one risk criterion in the target.** The Double-DQN target action must be chosen by the same criterion used for acting (CVaR or mean), and the thesis should state which. Terminal steps take target = r with no bootstrap.
- **Give the network the time remaining.** With γ = 1 over 20 steps, time-remaining must be a state feature.
- **Handle quantile crossing.** Sorting the predicted quantiles before computing CVaR is a cheap fix; disclose it if used. Monotone architectures exist (Théate et al., arXiv:2106.03228 **[V]**; Luo et al., ICLR 2022 **[V]**).
- **Treat quantiles as statistics.** Learned quantiles are statistics of the distribution, not samples from it (Rowland et al. 2019 **[V]** landing page; claim summary **[M]**). Do not, for example, bootstrap variance estimates from them as if they were draws.

**Statistical reporting.** Per agent, report test-set CVaR_α of total IS, the mean and the standard deviation, over ≥10 seeds, with stratified bootstrap confidence intervals and probability of improvement (Agarwal et al. 2021, [arXiv:2108.13264](https://arxiv.org/abs/2108.13264v4)) **[V]**. Use Colas et al. for seed power analysis ([arXiv:1806.08295](https://arxiv.org/abs/1806.08295v2)) **[V]** and Henderson et al. for the reasons seeds matter ([AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/11694/11553)) **[V]**.

**Tail estimates need many episodes.** At α = 0.1, 1,000 test episodes put only 100 in the tail, so report the CVaR estimator's own sampling error.

**Sanity checks against the Prasad audit.** Compare the learned initial-state quantiles with the empirical IS quantiles from simulator rollouts. Also include a risk-neutral QR-DDQN (mean action selection), so CVaR effects are not confused with distributional-learning effects.

## The adversarial shock-timing backup is open for execution, but not for market making

**The backup idea.** An adversary chooses the intra-episode timing and the depth-withdrawal size of a liquidity shock to maximise the execution agent's cost. It is evaluated on held-out shock timings and sizes, against uniform domain randomisation at an equal step budget. **No published execution paper matching this was found.**

**The lineage.**
- Robust adversarial RL (Pinto et al., ICML 2017, [PMLR](https://proceedings.mlr.press/v70/pinto17a.html)) **[V]**.
- Its trading application: Spooner & Savani's adversarial market maker (IJCAI 2020, [IJCAI](https://www.ijcai.org/proceedings/2020/633)) **[V]**. There the adversary sets drift and arrival/sensitivity parameters for each episode.

**The most dangerous prior art.** Yang & Xu (arXiv:2609.22785) **[V]** ([arXiv](https://arxiv.org/html/2609.22785v1)) extends this to Hawkes order flow and price impact. It evaluates with **CVaR at 10%** and includes a "Random" per-episode parameter-sampling baseline, which is effectively domain randomisation. So "adversary vs domain randomisation with CVaR evaluation" is not new in general.

**What still distinguishes the backup.**
- It is execution with a forced-completion constraint, not market making.
- The adversary acts *within* the episode on shock timing and depth, rather than setting episode-level parameters.

**Other neighbours, none of which fills the gap.**
- Cartea, Donnelly & Jaimungal (2017) **[V]**: robust control under model ambiguity, mostly for market making.
- Jaimungal et al. (2022) **[V]**: a Wasserstein-ball adversary for portfolios.
- ABIDES-MARL (Cheridito, Dupret & Wu 2025) **[V]**: emergent quasi-adversarial market makers, but no explicit adversary and no shocks.
- Olby et al. (ICAIF 2025) **[V]**: abstract only, which mentions no adversary or shocks.

**A wording caution.** The Hafsi & Vittori robustness section and the Olby et al. full text were not opened, so "not found" is the honest wording.

## Citations to double-check before the panel

**Could not be found, so do not cite.**
- **"Koh et al. 2023" (domain randomisation in finance)**: not found in any search. Replace it with a verified domain-randomisation reference or drop it. Within finance, Yang & Xu's "Random" baseline is the nearest verified analogue.

**Unverified [U].**
- Venue of Schubert et al. 2021 (likely an ICML workshop).
- Rowland et al. 2018 (AISTATS), not opened.
- Hau, Petrik & Ghavamzadeh's entropic-risk AISTATS 2023 paper, not searched.
- Volume and issue of Cheridito & Weiss in *Quantitative Finance* (DOI 10.1080/14697688.2026.2631116; the page failed to load).
- Authorship of the ResearchGate "RL for Trade Execution with Market Impact" (2025).
- arXiv:2602.09300, which was unreadable.
- Zakaria et al. (arXiv:2601.04896) is **withdrawn**: do not rely on it.
- Padnis (Stanford CS224R 2026) is a course project: cite it only as grey literature.

**From memory [M], to be checked against the publisher.**
- Ning, Lin & Jaimungal as *Applied Mathematical Finance* 28(4), 2021.
- Hendricks & Wilcox as IEEE CIFEr 2014.
- Karpe et al. as ICAIF 2020.
- Coache, Jaimungal & Cartea as *SIAM J. Financial Mathematics* 2023.
- Huang, Lehalle & Rosenbaum, *JASA* 110(509):107–122 (2015).
- A later *Applied Mathematical Finance* version of Macrì & Lillo.
- The algorithm and risk details of Nevmyvaka et al., Lin & Beling and Fang et al. (full texts not opened).
- Bao & Liu's use of DDPG.
- Lin & Beling's DQN liquidation workshop paper.
- Pricope 2021; TradeMaster (Sun et al. 2023).
- The *JFDS* volume of Cao et al. and the *Journal of Derivatives* placement of Halperin.
- Rockafellar–Uryasev volume and pages.
- Co-author lists for FQF, Rowland 2019, Keramati, DSAC, Henderson, and Byun et al.
- Spooner & Savani page range.

## Conclusion

**The proposal is stronger than its draft wording, but the claim should change.** It should not say the agent optimises CVaR. Its contribution is to measure whether a cheap, widely used CVaR heuristic actually thins the execution tail in a reactive market. In this setting the tail has an identifiable mechanical source: the forced end-of-horizon liquidation, which every prior paper imposes and none analyses.

That reframing turns the Lim & Malik caveat from a weakness into an experiment:
- **Hypothesis:** the CVaR-greedy agent and its threshold-augmented variant should both front-load selling, as Min et al.'s aggressive-in-the-money optimum predicts.
- **Test:** compare their terminal-trade shares of CVaR_α. A gap between them would measure, for the first time in execution, how much the per-step heuristic costs relative to the static objective.

The adversarial-shock backup remains a viable second contribution, provided Yang & Xu is cited as the market-making precedent.

## References

Tag shown after each entry: **[V]** checked against a primary or indexing page this session; **[M]** detail from memory; **[U]** unverified.

**Optimal execution with RL or CVaR**
- Bao, W. & Liu, X.-Y. (2019). Multi-Agent Deep Reinforcement Learning for Liquidation Strategy Analysis. arXiv:1906.11046. [V]
- Borjigin, A. & He, C. (2025). Safe and Compliant Cross-Market Trade Execution via Constrained RL and Zero-Knowledge Audits. arXiv:2510.04952. [V]
- Byun, W. J., Choi, B., Kim, S. & Jo, J. (2023). Practical Application of Deep Reinforcement Learning to Optimal Trade Execution. *FinTech* (MDPI) 2(3), article 23. [V; authors M]
- Cheridito, P., Dupret, J.-L. & Wu, Z. (2025). ABIDES-MARL: A Multi-Agent RL Environment for Endogenous Price Formation and Execution in a Limit Order Book. arXiv:2511.02016. [V]
- Cheridito, P. & Weiss, M. (2025/2026). Reinforcement Learning for Trade Execution with Market and Limit Orders. arXiv:2507.06345; *Quantitative Finance*, DOI 10.1080/14697688.2026.2631116. [V; volume U]
- de Witt, R. & Pakkanen, M. S. (2026). Diverse Approaches to Optimal Execution Schedule Generation. arXiv:2601.22113. [V]
- Espana, T., Hafsi, Y., Lillo, F. & Vittori, E. (2025). Reinforcement Learning in Queue-Reactive Models: Application to Optimal Execution. arXiv:2511.15262. [V]
- Fang, Y. et al. (2021). Universal Trading for Order Execution with Oracle Policy Distillation. AAAI 2021. [V listing; contents M]
- Hafsi, Y. & Vittori, E. (2024/2025). Optimal Execution with Reinforcement Learning in a Multi-Agent Market Simulator. arXiv:2411.06389. [V]
- Hendricks, D. & Wilcox, D. (2014). A reinforcement learning extension to the Almgren-Chriss framework for optimal trade execution. arXiv:1403.2229 (IEEE CIFEr 2014 [M]). [V]
- Karpe, M., Fang, J., Ma, Z. & Wang, C. (2020). Multi-Agent Reinforcement Learning in a Realistic Limit Order Book Market Simulation. arXiv:2006.05574 (ICAIF 2020 [M]). [V]
- Koulouris, C. S. & Campajola, C. (2026). Memory-Induced Supra-Competitive Outcomes Between Deep Reinforcement Learning Agents in Optimal Trade Execution. arXiv:2605.20348. [V]
- Leal, L., Laurière, M. & Lehalle, C.-A. (2022). Learning a functional control for high-frequency finance. *Quantitative Finance* 22(11):1973–1987. DOI 10.1080/14697688.2022.2106885; arXiv:2006.09611. [V]
- Lin, S. & Beling, P. A. (2020). An End-to-End Optimal Trade Execution Framework based on Proximal Policy Optimization. IJCAI 2020, paper 627. [V listing]
- Macrì, A. & Lillo, F. (2024). Reinforcement Learning for Optimal Execution when Liquidity is Time-Varying. arXiv:2402.12049; SSRN 4747466. [V]
- Micheli, A. & Monod, M. (2024). Deep Reinforcement Learning for Online Optimal Execution Strategies. arXiv:2410.13493. [V]
- Min, S., Moallemi, C. C. & Maglaras, C. (2022). Risk-Sensitive Optimal Execution via a Conditional Value-at-Risk Objective. Working paper (Columbia), rev. Jan 2022. [V]
- Moallemi, C. C. & Wang, M. (2022). A reinforcement learning approach to optimal execution. *Quantitative Finance* 22(6):1051–1069. DOI 10.1080/14697688.2022.2039403. [V]
- Nevmyvaka, Y., Feng, Y. & Kearns, M. (2006). Reinforcement Learning for Optimized Trade Execution. ICML 2006. [V listing]
- Ning, B., Lin, F. H. T. & Jaimungal, S. (2018/2021). Double Deep Q-Learning for Optimal Execution. arXiv:1812.06600 (*Applied Mathematical Finance* 28(4) [M]). [V]
- Olby, O., Bacalum, A., Baggott, R. & Stillman, N. (2025). Right Place, Right Time: Market Simulation-based RL for Execution Optimisation. ICAIF 2025. arXiv:2510.22206; DOI 10.1145/3768292.3770405. [V]
- Padnis, R. (2026). Alpha-Informed Optimal Trading Execution: RL with Domain Informed Priors. Stanford CS224R course project (grey literature). [V]
- Schnaubelt, M. (2022). Deep reinforcement learning for the optimal placement of cryptocurrency limit orders. *European Journal of Operational Research* 296(3):993–1006. DOI 10.1016/j.ejor.2021.04.050. [V]
- Zakaria, K. et al. (2026). Deep Reinforcement Learning for Optimum Order Execution: Mitigating Risk and Maximizing Returns. arXiv:2601.04896 (withdrawn). [V]

**Distributional RL**
- Bellemare, M. G., Dabney, W. & Munos, R. (2017). A Distributional Perspective on Reinforcement Learning. ICML 2017, PMLR 70. arXiv:1707.06887. [V]
- Bellemare, M. G., Dabney, W. & Rowland, M. (2023). *Distributional Reinforcement Learning*. MIT Press. ISBN 9780262048019. [V]
- Dabney, W., Ostrovski, G., Silver, D. & Munos, R. (2018). Implicit Quantile Networks for Distributional Reinforcement Learning. ICML 2018, PMLR 80. arXiv:1806.06923. [V]
- Dabney, W., Rowland, M., Bellemare, M. G. & Munos, R. (2018). Distributional Reinforcement Learning with Quantile Regression. AAAI 2018. arXiv:1710.10044. [V]
- Luo, Y. et al. (2022). Distributional Reinforcement Learning with Monotonic Splines. ICLR 2022. [V listing]
- Prasad, H. (2026). Auditing the Risk Claims of Distributional Reinforcement Learning. arXiv:2607.11607. [V]
- Rowland, M. et al. (2018). An Analysis of Categorical Distributional Reinforcement Learning. AISTATS 2018. arXiv:1802.08163. [U]
- Rowland, M., Dadashi, R., Kumar, S., Munos, R., Bellemare, M. G. & Dabney, W. (2019). Statistics and Samples in Distributional Reinforcement Learning. ICML 2019, PMLR 97. arXiv:1902.08102. [V; authors M]
- Théate, T., Wehenkel, A., Bolland, A., Louppe, G. & Ernst, D. (2023). Distributional Reinforcement Learning with Unconstrained Monotonic Neural Networks. *Neurocomputing*. arXiv:2106.03228. [V; volume U]
- Yang, D., Zhao, L., Lin, Z., Qin, T., Bian, J. & Liu, T.-Y. (2019). Fully Parameterized Quantile Function for Distributional Reinforcement Learning. NeurIPS 2019. arXiv:1911.02140. [V; authors M]

**CVaR and risk-sensitive RL**
- Bäuerle, N. & Ott, J. (2011). Markov Decision Processes with Average-Value-at-Risk criteria. *Mathematical Methods of Operations Research* 74(3):361–379. [V]
- Chow, Y. & Ghavamzadeh, M. (2014). Algorithms for CVaR Optimization in MDPs. NIPS 2014. arXiv:1406.3339. [V]
- Chow, Y., Tamar, A., Mannor, S. & Pavone, M. (2015). Risk-Sensitive and Robust Decision-Making: a CVaR Optimization Approach. NeurIPS 2015. arXiv:1506.02188. [V]
- Coache, A. & Jaimungal, S. (2024). Reinforcement learning with dynamic convex risk measures. *Mathematical Finance* 34(2):557–587. DOI 10.1111/mafi.12388; arXiv:2112.13414. [V]
- Coache, A., Jaimungal, S. & Cartea, Á. (2022/2023). Conditionally Elicitable Dynamic Risk Measures for Deep Reinforcement Learning. arXiv:2206.14666 (*SIAM J. Financial Mathematics* [M]). [V]
- Greenberg, I., Chow, Y., Ghavamzadeh, M. & Mannor, S. (2022). Efficient Risk-Averse Reinforcement Learning. NeurIPS 2022. arXiv:2205.05138. [V]
- Hau, J. L., Delage, E., Ghavamzadeh, M. & Petrik, M. (2023). On Dynamic Programming Decompositions of Static Risk Measures in Markov Decision Processes. NeurIPS 2023. arXiv:2304.12477. [V]
- Jaimungal, S., Pesenti, S. M., Wang, Y. S. & Tatsat, H. (2022). Robust Risk-Aware Reinforcement Learning. *SIAM Journal on Financial Mathematics*. DOI 10.1137/21M144640X; arXiv:2108.10403. [V]
- Kastner, T., Erdogdu, M. A. & Farahmand, A.-m. (2023). Distributional Model Equivalence for Risk-Sensitive Reinforcement Learning. NeurIPS 2023. arXiv:2307.01708. [V]
- Keramati, R., Dann, C., Tamkin, A. & Brunskill, E. (2020). Being Optimistic to Be Conservative: Quickly Learning a CVaR Policy. AAAI 2020, 34(04). arXiv:1911.01546. [V; authors M]
- Lim, S. H. & Malik, I. (2022). Distributional Reinforcement Learning for Risk-Sensitive Policies. NeurIPS 2022 (Advances in NeurIPS 35). [V]
- Ma, X., Xia, L., Zhou, Z., Yang, J. & Zhao, Q. (2020/2025). DSAC: Distributional Soft Actor-Critic for Risk-Sensitive Reinforcement Learning. arXiv:2004.14547; *JAIR* 2025. [V; authors M]
- Rockafellar, R. T. & Uryasev, S. (2000). Optimization of conditional value-at-risk. *Journal of Risk* 2(3):21–41. [V; pages M]
- Schubert, F. et al. (2021). Automatic Risk Adaptation in Distributional Reinforcement Learning. arXiv:2106.06317. [venue U]
- Tamar, A., Glassner, Y. & Mannor, S. (2015). Optimizing the CVaR via Sampling. AAAI 2015. arXiv:1404.3862. [V]
- Urpí, N. A., Curi, S. & Krause, A. (2021). Risk-Averse Offline Reinforcement Learning. ICLR 2021. arXiv:2102.05371. [V]

**Adjacent finance, robustness and simulators**
- Buehler, H., Gonon, L., Teichmann, J. & Wood, B. (2019). Deep hedging. *Quantitative Finance* 19(8):1271–1291. arXiv:1802.03042. [V]
- Cao, J., Chen, J., Hull, J. & Poulos, Z. (2021). Deep Hedging of Derivatives Using Reinforcement Learning. *Journal of Financial Data Science* 3(1). arXiv:2103.16409. [V; volume M]
- Cartea, Á., Donnelly, R. & Jaimungal, S. (2017). Algorithmic trading with model uncertainty. *SIAM J. Financial Mathematics* 8(1):635–671. DOI 10.1137/16M106282X. [V]
- Halperin, I. (2017/2020). QLBS: Q-Learner in the Black-Scholes(-Merton) Worlds. arXiv:1712.04609 (*Journal of Derivatives* [M]). [V]
- Hambly, B., Xu, R. & Yang, H. (2023). Recent Advances in Reinforcement Learning in Finance. *Mathematical Finance* 33(3):437–503. arXiv:2112.04553. [V listing]
- Huang, W., Lehalle, C.-A. & Rosenbaum, M. (2015). Simulating and Analyzing Order Book Data: The Queue-Reactive Model. *JASA* 110(509):107–122. arXiv:1312.0563. [M]
- Pinto, L., Davidson, J., Sukthankar, R. & Gupta, A. (2017). Robust Adversarial Reinforcement Learning. ICML 2017, PMLR 70. arXiv:1703.02702. [V]
- Spooner, T., Fearnley, J., Savani, R. & Koukorinis, A. (2018). Market Making via Reinforcement Learning. AAMAS 2018. arXiv:1804.04216. [V]
- Spooner, T. & Savani, R. (2020). Robust Market Making via Adversarial Reinforcement Learning. IJCAI 2020, paper 633. arXiv:2003.01820. [V]
- Yang, H. & Xu, Z. (2026). Robust Market Making with Hawkes Order Flow and Price Impact via Adversarial Reinforcement Learning. arXiv:2609.22785. [V]

**Evaluation methodology**
- Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A. & Bellemare, M. G. (2021). Deep Reinforcement Learning at the Edge of the Statistical Precipice. NeurIPS 2021. arXiv:2108.13264. [V]
- Colas, C., Sigaud, O. & Oudeyer, P.-Y. (2018). How Many Random Seeds? Statistical Power Analysis in Deep Reinforcement Learning Experiments. arXiv:1806.08295. [V]
- Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D. & Meger, D. (2018). Deep Reinforcement Learning that Matters. AAAI 2018. arXiv:1709.06560. [V; authors M]
- Patterson, A., Neumann, S., White, M. & White, A. (2024). Empirical Design in Reinforcement Learning. *JMLR* 25 (23-0183). arXiv:2304.01315. [V]
