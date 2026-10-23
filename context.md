# Project context: CC-PPO adversarial crypto trading (BTech final year project)

This is the shared design reference for the repository. Read it alongside
`ppo/README.md` for setup and `docs/experiment-notes.md` for dated results.
Automatic loading by an assistant depends on that assistant's configuration.

## Team
4 members, BTech CSE final year project. ~6-7 month timeline.

## Current implementation (final BTC configuration)

`ppo/src/coevo_node.py` selects the final BTC run; `EngineConfig` defaults and
legacy V4–V6 scripts describe earlier variants. The current settings are:

- Three arms: single PPO, population + PBT, and population + PBT with an
  evolving environment population. PBT evolves learning rate, entropy
  coefficient, and clipping range; PPO updates network weights.
- Six seeds per named run, 12 generations, six traders for population arms
  (one for single), and 12 adversaries in the coevolution arm.
- Five market inputs: `volatility_168h`, `ema50_ema200_ratio`, and cross-fitted
  ridge forecasts `rhat4`, `rhat24`, `rhat72`, plus three account values.
  The shared 19-feature base schema is in `feature_columns.py`; explicit
  node inputs override the base observation list.
- Two target-position actions: cash or long, every four hourly bars. Episodes
  start invested. Costs are maker fee 0.02%, slippage 0.01%, and funding;
  the spot buy-and-hold benchmark pays 0.1% plus 0.05% slippage.
- In coevolution, half of each training match's segments come from the adversary. Regime lock
  preserves the uniform pool's regime mix while the adversary selects within
  each regime. An eight-segment shared exam ranks coevolution traders on common
  windows; the other arms retain their match-based fitness.
- Adversaries use absolute learning progress: change in population-mean
  performance on a window since its previous exam. Other engine objectives
  remain available for historical experiments.
- Final populations vote by the mean action probabilities, then results are
  averaged across seeds. The headline pools four runs, with and without
  bounded observation drift; the run matrix is in `ppo/README.md`.

The six development folds end before 2026-05-01 07:00 UTC. Earlier fold tests
can enter later training windows. May–September 2026 was used in historical
holdout/prospective checks and is spent; it is not fresh evaluation data.
The final BTC headline is a development result. October 12 conclusions in
the experiment notes refer to earlier configurations and cost contracts.

The BTC evaluator uses paired segment bootstrap probabilities. The legacy
`compare_arms.py` also reports seed and hierarchical bootstraps; the final
evaluator does not automatically run a deflated-Sharpe or reality-check test.
Forecast training inputs are cross-fitted across blocks, including later
training blocks; external test forecasts use purged training labels.

Current limitations: learning-progress history is not checkpointed, so resume
resets that history. Synthetic benchmarks use named versions and saved
parameters; source-file hash metadata and checks have been removed. Changing
benchmark parameters should be treated as a new experiment.

## The core question and why it matters
The real risk with RL-for-trading projects is not "can we show a profit" (trivial
in a naive backtest, and almost always fake — lookahead bias, unrealistic costs,
single lucky seed, testing on a bull-only window, or the agent collapsing to a
degenerate policy like "always long" that just rides market beta). The real risk
is producing a paper that *looks* rigorous but doesn't survive scrutiny.

**Our hypothesis is explicitly NOT "the agent makes money."** It is:

> Competitive coevolutionary training produces PPO trading policies that
> generalize better to unseen and out-of-regime market conditions — smaller
> out-of-sample degradation and better risk-adjusted metrics — than an
> equivalently-tuned single-agent PPO baseline.

This is allowed to resolve as yes, no, or mixed. The grade/contribution comes
from methodological rigor (real baselines, walk-forward evaluation, multiple
seeds, honest failure-mode analysis), not from finding a money printer. A
rigorous negative or mixed result is a legitimate, strong outcome. Chasing raw
profit is what causes these projects to quietly overfit and collapse.

## What "CC" means here
CC = **competitive coevolutionary**. Two populations, coupled fitness, coevolving
each generation:

- **Trader population (agent side):** a population of PPO trading agents.
- **Environment population (environment side):** a population of adversarial
  "scenario generators" — informally the "adversary," but we standardize on
  **environment population** in writing (define both terms once, then use one
  consistently). Each individual is a segment-selection strategy: a small
  parameter vector (preference weights over segment features like volatility
  percentile, trend strength, drawdown depth, regime-transition presence) that
  picks which real historical windows from the data pool the trader has to
  trade on.

### Why the environment population only *selects* real historical segments (critical constraint)
An unconstrained adversary that can invent arbitrary price paths will construct
impossible markets (e.g. a straight line to zero) that no strategy could survive
— the trader either learns nothing or collapses into a uselessly conservative
policy, and the whole arms race degenerates. By restricting the environment
population to **selecting from real historical segments** rather than
fabricating prices, every scenario is automatically plausible because it
actually happened. This is the single most important design constraint in the
project — do not relax it without a very good reason and a plan to handle the
consequences.

The implemented optional perturbation changes selected observation inputs by
bounded shifts and scales; it does not fabricate price paths. Bounded price
perturbations remain an unimplemented proposal. Synthetic benchmarks provide
separate, explicitly generated datasets rather than modifying BTC prices.

## How evolutionary operators apply differently to the two populations
This is a common point of confusion — the two populations are different kinds
of objects and the operators mean different things on each.

- **Environment population:** crossover + mutation apply directly and cleanly.
  Segment selection is non-differentiable (can't backprop through "which
  historical window did I pick"), so evolutionary search is the natural
  optimizer. Crossover blends two individuals' preference vectors; mutation
  perturbs them.
- **Trader population:** do NOT apply crossover/mutation to PPO network
  weights directly. Weight-space crossover is destructive (permutation
  symmetry — two networks can compute the same function with neurons in
  different order, so blending weights usually breaks both), and weight
  mutation competes with the PPO gradient rather than complementing it.
  Instead, evolve **hyperparameters** via Population-Based Training (PBT):
  crossover/mutation act on learning rate, entropy coefficient, and clip range.
  Reward shaping and lookback evolution are proposals, not implemented PBT dimensions. PPO's gradient still does all
  the actual weight learning inside each individual.

## Known failure modes of two-sided coevolution (guard against these from day one)
Evolving both populations is the "real" coevolutionary regime and is more
scientifically interesting, but it is also less stable. Build these guardrails
in from the start, don't retrofit them:

1. **Disengagement** — if one side gets far ahead (adversaries too strong, or
   trader too strong), the other side's fitness landscape goes flat and
   learning stalls. Mitigate with asymmetric variation rates (tune so neither
   side runs away) and a hall of fame (see below).
2. **Red Queen dynamics** — internal relative fitness looks like improvement
   while real, absolute performance is static. Mitigate by tracking an
   **external, objective yardstick** (fixed baselines + truly held-out data)
   completely separately from the in-population relative fitness. Report both.
3. **Cycling / intransitivity** — trader A beats environment-individual B beats
   trader C beats environment-individual A, endlessly, with no real progress.
   Mitigate with a **hall of fame / archive**: keep past individuals from both
   populations and evaluate against them too, so nothing can "forget" how to
   beat old opponents. This is the single most effective anti-cycling fix.
   **Implementation trap (found in the Sep 2026 audit, fixed in V5):** SB3's
   `get_parameters()` returns dicts that alias the live tensors. Archiving
   them without a deep copy makes the "snapshot" follow the live trader, and
   loading an archive entry into a borrowed live model overwrites that model.
   V1–V4's trader hall of fame had both defects, so in the coevolution arm
   trader 0 was cloned from another live trader every generation and the
   adversary-side archive was inert. `hall_of_fame.snapshot_parameters` and
   the engine's dedicated shell network are the fix; the smoke test asserts
   both.
4. **Diversity collapse** on the environment side — all individuals converging
   on the same single exploit. Mitigate with diversity pressure / competitive
   fitness sharing so environment individuals are rewarded for being difficult
   in *different* ways.

## Terminology (keep consistent across the whole team and the paper)
- **PnL** = profit and loss, the trader's return on a segment. Should be
  reported **net of transaction costs and slippage**, and primarily as a
  risk-adjusted metric (Sharpe etc.), not raw return — a strategy that made
  money by taking excessive risk should not look good.
- **Adversary** vs **environment population** — same thing. Introduce once as
  "environment population (the adversary)," then pick one term and use it
  consistently for the rest of the document/codebase.

## Pipeline (five stages)

1. **Data processing** (offline, deterministic, done once):
   Raw OHLCV (single coin, one timeframe) → cleaning + feature engineering
   (indicators, regime labels: bull/bear/sideways) → walk-forward split
   producing a **pool of real historical segments**. This pool is what the
   environment population later draws from.

2. **Market simulator (gym environment):**
   Wraps the segment pool. Defines observation (current feature row + portfolio
   state), action space, and reward net of transaction costs and slippage.
   Parameterized by which segment it's given — this is how the environment
   population exerts control.

3. **Competitive coevolutionary engine (the core contribution):**
   - Trader population (PPO agents, agent side) and environment population
     (segment-selection strategies, environment side) interact through the
     simulator each generation.
   - Environment population selects windows → traders train and are examined →
     the configured objective supplies adversary fitness. The original objective
     negates trader fitness; the final BTC node uses learning progress.
   - With a shared exam, traders are ranked on the same sampled windows before
     PBT. Optional observation drift applies only to configured inputs.
   - Selection + variation (crossover + mutation) applied to **both**
     populations each generation — PBT-style hyperparameter evolution for the
     trader, direct genome crossover/mutation for the environment population.

4. **Evaluation harness:**
   Held-out, non-adversarial real regimes. Fixed baselines: buy-and-hold,
   MA-crossover, single-agent PPO (no coevolution). Risk-adjusted metrics:
   Sharpe, Sortino, max drawdown, Calmar. Multiple seeds, variance reported.
   Paired bootstrap comparisons quantify uncertainty under the selected
   resampling scheme. Development-time model selection remains a limitation;
   the historical ORX deflated-Sharpe calculation is a separate reported analysis. This stage is deliberately
   **outside** the coevolutionary arms race — it answers "did this generalize
   to real unseen data," not "did the trader beat its own adversary" (which
   proves nothing). The external-objective feedback line from here back into
   the engine is what guards against Red Queen dynamics (see failure modes).

5. **Prospective validation (original proposal):**
   This stage is not an automated workflow in the committed code. The historical
   September 2026 check in the experiment notes used data downloaded later.
   The original proposal was to freeze a model, then forward paper-trade for ~4-8 weeks on data that didn't
   exist when the model was frozen (Binance testnet is fine for the demo here,
   or live mainnet data paper-traded). This is the strongest form of
   out-of-sample evidence when model and selection rules stay frozen before
   the new observations arrive.
   Do NOT make testnet the main evidence — testnet liquidity/fills aren't real
   and it's non-reproducible. The historical walk-forward backtest (stage 4) is
   the real evidentiary backbone; prospective validation is the compelling
   add-on.

## Evaluation design non-negotiables
- Multiple seeds always — PPO is high-variance across seeds; a single run
  proves nothing.
- Walk-forward / rolling out-of-sample across **multiple regimes** (bull,
  bear, sideways) — not a single train/test split.
- Realistic transaction costs and slippage modeled from the start, not added
  later.
- Real baselines every time: buy-and-hold, MA-crossover/momentum, and
  single-agent PPO without coevolution.
- Report risk-adjusted metrics, not just raw return.

## Ablations that ARE the paper's argument
Single-agent PPO baseline vs. population-only (no adversary) vs. full
adversarial coevolution (both populations evolving). These three comparisons
are the core empirical contribution — design experiments explicitly around
them.

## Original proposed milestone plan (~24 weeks)
This schedule records the initial planning assumptions, not a live task list.
Implemented behavior is summarized above; dated experiments are in the notes.

- **Weeks 1-3 — Foundations:** data pipeline, gym environment with realistic
  costs/slippage, regime labeling, reproducibility harness (seeding, config
  management, logging e.g. Weights & Biases, walk-forward split scaffolding).
- **Weeks 4-7 — Single PPO baseline (the control):** beat buy-and-hold and
  MA-crossover on walk-forward backtest, multiple seeds. Submittable on its
  own; de-risks the whole project.
- **Weeks 8-12 — Coevolutionary scaffolding:** trader population with
  tournament/relative fitness, selection + variation. Track internal relative
  fitness AND external objective performance separately from the start; watch
  for Red Queen cycling early.
- **Weeks 13-18 — Environment population / core contribution:** build the
  bounded environment population (segment selection first), run the full
  arms race, run the three-way ablation.
- **Weeks 19-24 — Validation, analysis, writing:** prospective paper-trading
  on frozen models (start this well before the deadline — it runs in real
  wall-clock time), regime-robustness analysis, environment-population
  behavior analysis, failure-mode characterization, write-up.

**Scheduling note:** freeze a "v1" model around month 4-5 so the 4-8 week
prospective validation window elapses while the team writes — don't leave it
until the end.

## Original proposed team role split
1. Data & environment: pipeline, feature engineering, gym env, cost/slippage
   model, regime labels.
2. RL / trader: PPO implementation and tuning, single-agent baseline, trader
   population + PBT.
3. Coevolution & environment population: population dynamics, selection,
   environment population design, the arms-race loop, hall of fame.
4. Evaluation & reproducibility: metrics suite, walk-forward harness,
   statistical significance, logging, prospective validation, figures.
