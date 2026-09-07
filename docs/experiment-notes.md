# CC-PPO — Experiment Notes & Reference

A from-scratch reference for the competitive coevolutionary PPO (CC-PPO) crypto-trading
project: what was done, how the two evolution loops work in code, the terminology, the
metrics, and the first results. Written to be readable with **no prior background**.

- Design doc (authoritative): [`../context.md`](../context.md)
- Design history: [`design-history.md`](design-history.md)
- Code: [`../ppo/src/`](../ppo/src/)

---

## Table of contents
1. [Big picture — what and why](#1-big-picture)
2. [Glossary of every term](#2-glossary)
3. [The pipeline, step by step](#3-pipeline)
4. [How the adversary (environment population) evolves](#4-adversary-evolution)
5. [How the traders (PPO agents) evolve](#5-trader-evolution)
6. [How the two loops connect each generation](#6-generation-loop)
7. [The metrics (Sharpe, Return %, Max DD, Calmar)](#7-metrics)
8. [Fitness function — what you can/should use](#8-fitness)
9. [First results & the two failure modes](#9-results)
10. [Next steps](#10-next-steps)

---

<a name="1-big-picture"></a>
## 1. Big picture — what and why

We train an AI to trade Bitcoin, but **the goal is not profit** — it is a science question:

> Does **competitive coevolutionary training** produce PPO trading agents that
> **generalize better** to unseen / out-of-regime market conditions than a normally
> trained single agent?

The contribution is **methodological rigor** (real baselines, multiple seeds, walk-forward
out-of-sample testing, honest failure analysis), not finding a money printer. A rigorous
negative or mixed result is still a valid outcome.

---

<a name="2-glossary"></a>
## 2. Glossary of every term

| Term | Plain meaning |
|---|---|
| **Reinforcement Learning (RL)** | Training by trial-and-error: agent takes an **action**, gets a **reward**, learns to act to maximize reward. |
| **Agent** | The AI decision-maker. Here it chooses **buy / sell / hold** each hour. |
| **PPO (Proximal Policy Optimization)** | A specific, stable RL algorithm — the "recipe" the agent learns with. Provided by the `stable-baselines3` library. |
| **Environment** | The simulated world the agent acts in — here, a replay of real hourly BTC prices with realistic fees + slippage. Code: `trading_env.py`. |
| **Observation** | What the agent sees each step: 19 market indicators + 3 account numbers (position, cash %, asset %). |
| **Action space** | The choices available: 0 = hold, 1 = buy, 2 = sell (discrete, size 3). |
| **Reward** | Signal the agent maximizes: the **log return** of the portfolio each step (grows the account). |
| **Fee / slippage** | Realistic trading costs (0.1% fee, 0.05% slippage) charged in the sim so results aren't fake. |
| **Feature / technical indicator** | A derived number summarizing market state (RSI, MACD, moving averages, volatility, Bollinger Bands…). 19 of them feed the agent. |
| **Segment** | A fixed-length window of real historical price data (e.g. ~30 days). The unit the adversary selects. |
| **Regime** | The market's character in a window: **bull** (rising), **bear** (falling), **sideways** (flat). Labeled without lookahead in `regimes.py`. |
| **Coevolution (competitive coevolutionary)** | Training **two opposing populations** together — traders vs. an adversary — so they push each other (an "arms race"). |
| **Trader population** | A group of PPO agents evolving together. |
| **Environment population / adversary** | A group of small "scenario-picker" individuals that choose the **hardest real windows** to give the traders. Can only *select* real history, never invent prices. |
| **Single (mode)** | Control setup: one lone PPO agent, random windows, no evolution. "The normal way." |
| **Population-only (mode)** | A trader population that evolves, but **no adversary** (random windows). Isolates the effect of "having a population." |
| **Coevolution (mode)** | The full method: trader population **vs.** evolving adversary. |
| **Ablation** | A comparison that turns features on/off one at a time to see which one actually matters. The single / population / coevolution comparison **is** the paper's argument. |
| **Baseline** | A simple non-AI reference strategy the AI must beat to prove its worth. |
| **Buy & Hold** | Buy once at the start, never sell. Hard to beat in a rising market. |
| **MA-crossover** | Rule-based strategy: buy when the 20-hour average price rises above the 50-hour average, sell when it falls below. Classic momentum benchmark. |
| **Seed** | A number that makes randomness reproducible — same seed → same run. We use 42, 43, 44. |
| **Why 42?** | Long-running programmer joke (*Hitchhiker's Guide*: "answer to everything"), the de-facto default example seed in ML. 43/44 are just the next two — three *different* runs to average over. Any three distinct numbers would do. |
| **Why 3 seeds?** | PPO is high-variance; one run could be luck. Averaging over ≥3 seeds (and reporting spread) shows whether a result is real. |
| **Generation** | One round of the population improving (train → score → evolve). Runs default to 20 generations. |
| **PBT (Population-Based Training)** | The technique used to evolve the traders' *hyperparameters* (not their weights) — see §5. |
| **Hyperparameter** | A setting controlling *how* an agent learns (learning rate, exploration, etc.), as opposed to *what* it learned (the weights). |
| **Hall of Fame** | An archive of strong past individuals; agents are re-tested against them so nothing "forgets" how to beat old opponents (anti-cycling). |
| **Fitness** | The score used to rank individuals for selection (traders: their trading metric; adversary: the *inverted* trader score). |
| **Fitness sharing** | Penalizing individuals that are too similar, so the population stays diverse. |
| **In-distribution / out-of-regime** | "Like the training data" vs. "a market character not seen in training." Generalizing to out-of-regime data is the whole point. |
| **Train / validation / test split** | Three time-separated data chunks: learn on train, tune on val, final exam on test (never touched until the end). |

---

<a name="3-pipeline"></a>
## 3. The pipeline, step by step

### Stage 0 — Get the data
Downloaded **107 monthly Binance CSV files** of **BTCUSDT** (Bitcoin/USD), **1-hour candles**,
**Aug 2017 → Jun 2026**, into `ppo/data/raw/`. Each row = one hour of Open/High/Low/Close/Volume
("OHLCV").

### Stage 1 — Clean (`data_collect.ipynb` → `data/cleaned/`)
Merge all files, fix timestamps, drop duplicates, sort chronologically → **77,629 clean hourly candles**.

> **Bug fixed:** Binance switched timestamps from **milliseconds (13 digits)** to
> **microseconds (16 digits)** in Jan 2025. Old notebook code crashed on the newer files
> (`OutOfBoundsDatetime`). Patched cell 1 to leave `open_time` as a raw integer and let the
> `normalize_timestamp()` step convert both eras to a single unit before building dates.

### Stage 2 — Features (`feature_engineering.ipynb` → `data/features/`)
Compute **19 technical indicators** from raw prices (RSI, MACD, EMAs, ATR, volatility,
Bollinger Bands, volume ratios…). These become the agent's observation. Output had **0 NaNs**.

### Stage 3 — Split + scale (`train_test.ipynb` → `data/processed/`)
Time-ordered, non-overlapping split (no peeking at the future):

| Split | Period | Rows | Character |
|---|---|---|---|
| **Train** | before 2025 | 64,353 | mixed |
| **Validation** | Jan–Oct 2025 | 6,552 | **bull** (+20.8%) |
| **Test** | Oct 2025 → | 6,552 | **bear + sideways, no bull** |

Features are **standardized** (StandardScaler) fit on **train only**; scaler saved to
`models/feature_scaler.pkl`. The bull-val / bear-test contrast makes this a genuine
**out-of-regime generalization test**.

> **Env fixed:** `ppo/.venv` was missing `ta`, `openpyxl`, `tensorboard` (listed in
> requirements but not installed). Installed. **Use `ppo/.venv`, not the root `venv`.**

### Stage 4 — Single-agent baseline (`train_ppo.py`)
Train the plain PPO agent, 3 seeds, evaluate on val. Result: Sharpe swung **−0.52 / +0.89 / +0.65**
(mean 0.34) — **high variance**, did not reliably beat Buy & Hold (0.80) on the bull window,
but took much less risk. The variance itself motivates the robustness question.

### Stage 5 — Three-way ablation (`run_coevolution.py`)
Run all three modes × 3 seeds = **9 runs**, 20 generations each. Outputs per run in
`runs/<mode>_seed<s>/`: `config.json`, `generations.csv` (per-generation log), `best_trader.zip`.

### Stage 6 — Evaluate on the test set (`evaluate.py`)
Take each mode's frozen models, evaluate on the **untouched test data** per regime, against
Buy & Hold and MA-crossover on identical windows, with a **paired-bootstrap significance test**.

---

<a name="4-adversary-evolution"></a>
## 4. How the adversary (environment population) evolves

**File:** `ppo/src/coevo/env_population.py`

### What one adversary is
Not a neural network — a **tiny number-vector**:
```python
@dataclass
class EnvIndividual:
    weights: np.ndarray   # 5 numbers: preference over segment "descriptors"
    temperature: float    # how picky it is
```
Each historical window is summarized by **5 descriptors** (from `segments.py`):
`volatility_pct, trend_strength, trend_direction, drawdown_depth, regime_transitions`.
So `weights` = *what kind of window this adversary likes to inflict* (e.g. high-volatility,
deep-drawdown, downward windows).

### How it picks windows
Scores every segment as `descriptor · weights`, softmaxes into probabilities, samples a few:
```python
logits = descriptor_matrix @ self.weights / self.temperature
p = softmax(logits)         # high prob to windows matching its preferences
```
`temperature` low → almost always its favorite nasty windows; high → more random.
**Hard constraint:** it can only *select real historical windows*, never fabricate prices.

### How it's scored
The adversary wins when the trader loses, so it records the **negated** trader score:
```python
def record(self, trader_score):
    self._scores.append(-trader_score)
```

### The evolution (genetic algorithm — `evolve()`)
Because an adversary is just numbers, classic GA operators apply directly. Each generation:
1. **Rank** adversaries by fitness.
2. **Elitism** — copy the top ~25% unchanged into the next generation.
3. **Fill the rest with offspring**, each from:
   - **Tournament selection** — pick a few at random, keep the best; do twice → two parents.
   - **Crossover** — blend the two parents' weight vectors: `alpha*a + (1-alpha)*b` per dimension.
   - **Mutation** — add Gaussian noise to weights (clip ±3); jitter temperature.
4. **Fitness sharing** (`apply_fitness_sharing`) — penalize adversaries whose window-preference
   distributions overlap too much, so they stay diverse (rewarded for being hard in *different*
   ways, not all converging on one exploit).

**Summary:** rank → keep elites → breed the rest via tournament + crossover + mutation, with
diversity pressure. Adversaries that reliably find windows the traders fail on get selected.

---

<a name="5-trader-evolution"></a>
## 5. How the traders (PPO agents) evolve

**File:** `ppo/src/coevo/trader_population.py`

A trader **is a neural network**. You **cannot** blend two networks' weights — two nets can
compute the same function with neurons in different orders, so averaging scrambles both. So the
trader side is evolved very differently from the adversary.

### Two separate improvement mechanisms
- **(a) Weights learn by PPO gradient descent — NOT evolution.** Each trader trades its windows
  and PPO updates its weights (`train_on_segments` → `model.learn(...)`). Evolution never edits
  weights directly.
- **(b) Hyperparameters evolve — via PBT.** We evolve three settings that control *how* it learns:
  ```python
  HYPERPARAM_BOUNDS = {
      "learning_rate": (1e-5, 1e-3),  # step size
      "ent_coef":      (1e-4, 0.1),   # exploration vs exploitation
      "clip_range":    (0.05, 0.4),   # how cautious each PPO update is
  }
  ```

### PBT — Population-Based Training (`pbt_step`)
Once per generation:
1. **Rank** traders by fitness.
2. Bottom group = "losers", top group = winners.
3. Each loser:
   - **Exploit** — copy an entire winning network **wholesale** (copied, *not* blended — that's
     the key safety point vs. the adversary side):
     ```python
     loser.model.set_parameters(parent_a.model.get_parameters())
     ```
   - **Explore** — take hyperparameters from a **crossover** of two winners, then **mutate**:
     ```python
     loser.hyperparams = mutate_hyperparams(
         crossover_hyperparams(parent_a.hyperparams, parent_b.hyperparams))
     ```
     - crossover: per setting, randomly take winner A's or B's value.
     - mutate: multiply each setting by 1.25 or ÷1.25, clipped to bounds.

**Intuition:** a continuous talent competition. Weak agents are replaced by a **copy of a winner**
(no wasted effort), but each copy gets **slightly different learning settings** and keeps training —
so the population explores many learning configurations at once and good settings spread. **PPO does
the weight learning; PBT steers which configurations survive.**

### The contrast in one line
- **Adversary** = number-vectors → **crossover + mutation on the genes directly** (a GA).
- **Trader** = neural nets → **PPO gradient learns the weights; PBT copies whole winners + evolves hyperparameters.**

---

<a name="6-generation-loop"></a>
## 6. How the two loops connect each generation

**File:** `ppo/src/coevo/engine.py` (`run`). Each generation:
1. **Matches** — each trader is handed windows *chosen by an adversary* (coevolution mode) or
   random (other modes), trains on them (PPO), and is scored. Adversary records the inverted score.
   Extra matches vs. Hall-of-Fame archives guard against "forgetting."
2. **Finalize fitness** for both sides; apply adversary fitness sharing.
3. **Log two ledgers separately:**
   - **internal** — who's winning the arms race (proves nothing alone).
   - **external** — the best trader tested on fixed held-out data vs. fixed baselines (the honest
     yardstick). Kept separate to catch "Red Queen" illusions.
4. **Evolve both** — `ep.evolve(...)` for adversaries, `tp.pbt_step(...)` for traders.

Round after round: adversaries get better at picking painful windows → traders get better at
surviving them → repeat. That mutual pressure is the **arms race**.

---

<a name="7-metrics"></a>
## 7. The metrics

**File:** `ppo/src/metrics.py`. All computed from **portfolio value over time**; **returns** are the
hour-to-hour % changes.

### Return % — "how much did I make?"
```
return_% = (final_value / initial_value − 1) × 100
```
Simple, but misleading alone — big returns can come from reckless risk.

### Sharpe ratio — "return per unit of bumpiness" (headline metric)
```
sharpe = mean(returns) / std(returns) × sqrt(periods_per_year)     # periods_per_year = 24×365
```
Reward relative to volatility. High = smooth, quality returns; negative = erratic or losing.
`24×365` annualizes hourly data so it matches standard finance quoting. **This is why coevolution
(−1.04) beat single (−1.64) despite both losing — it lost more smoothly / with less risk.**

### Max Drawdown (Max DD) — "worst drop I sat through"
```
drawdown  = (value − running_max) / running_max       # ≤ 0 always
max_dd    = the most negative drawdown
```
Smaller (closer to 0) = safer. Coevolution's standout: **−5.3% vs Buy & Hold −16.2%.**

### Calmar ratio — "return per unit of worst drop"
```
calmar = annualized_return / |max_drawdown|
```
Higher = better reward for the worst risk taken. **Noisy** when drawdown ≈ 0 (division blows up) —
saw ±7 std in the results.

### Sortino (bonus, also in code)
Like Sharpe but only penalizes **downside** volatility (upside swings aren't treated as "risk").

**Why four, not one?** Each is gameable alone (return hides risk; Sharpe can come from barely
trading). Reporting all four makes it hard to look good by accident.

---

<a name="8-fitness"></a>
## 8. Fitness function — what you can / should use

The engine already supports **any single metric** as fitness via `EngineConfig.fitness_metric`
(`evaluate_on_segments` computes all metrics, then selects `m[fitness_metric]`). Valid values:
`total_return_pct`, `sharpe`, `sortino`, `max_drawdown_pct`, `calmar`.

| Candidate | Use as fitness? | Why |
|---|---|---|
| **Sharpe** (default) | ✅ Best default | Risk-adjusted; not gameable by reckless bets. |
| **Sortino** | ✅ Good alternative | Penalizes only downside. |
| **Calmar** | ⚠️ Noisy | Divides by drawdown → unstable near 0 (saw ±7 std). |
| **Return %** | ❌ Avoid alone | Rewards risk-taking; degenerates to "always long." |
| **Max DD** | ❌ Avoid alone | Rewards *never trading* (0 drawdown = "perfect"). |

**Rules of thumb:**
- Keep a risk-adjusted metric (Sharpe/Sortino) as fitness.
- A **weighted composite** (e.g. `0.7·Sharpe − 0.3·|MaxDD|`) is possible but needs a code change
  **and** scale-normalization (the metrics live on different ranges). Multi-objective/Pareto is the
  cleaner research version.
- **Keep evaluation separate from fitness** — still report all four at eval time, and ideally don't
  judge success *only* by the exact metric you optimized (slightly circular).
- Swapping the fitness metric is itself a legitimate mini-ablation for the paper.

---

<a name="9-results"></a>
## 9. Results & the two failure modes

Three runs exist:
- **V1** (`runs/`) — 3 seeds, all defaults.
- **V2** (`runs/v2/`) — 6 seeds (42–47), tuned adversary variation
  (`env-weight-σ 0.7`, `env-temp-σ 0.35`, `n-env 16`, `segments-per-match 6`).
- **V3** (`runs/v3/`) — **the headline.** V2 settings plus the diagnostic-driven fixes:
  `trade-penalty 0.003`, `gamma 0.997`, `ent-coef-max 0.02`, `env-fresh-frac 0.25`.

**V3 test set (Oct 2025 – Jun 2026; 9 segments: 6 sideways, 3 bear; 6 seeds; mean ± std):**

| Arm | Sharpe | Return % | Max DD % | P(≤ B&H) | Beats B&H? |
|---|---|---|---|---|---|
| **Coevolution** | **−0.38 ± 0.32** | **−1.81** | **−5.57** | **0.131** | ✅ |
| Single (control) | −1.27 ± 0.46 | −4.93 | −12.56 | 0.479 | ❌ |
| Population-only | −1.49 ± 0.21 | −6.14 | −15.28 | 0.874 | ❌ |
| Buy & Hold | −1.30 | −6.33 | −16.20 | — | — |
| MA-crossover | −0.47 | −1.51 | −7.72 | — | — |

**Progression across the three runs (coevolution arm):**

| | V1 (3 seeds) | V2 (6 seeds) | **V3 (6 seeds + fixes)** |
|---|---|---|---|
| Sharpe | −1.04 ± 0.75 | −0.65 ± 0.76 | **−0.38 ± 0.32** |
| Seed spread (std) | 0.75 | 0.76 | **0.32** (halved) |
| P(≤ B&H) | 0.393 | 0.244 | **0.131** |

**Reading it:**
- **The fixes worked.** Best Sharpe yet, and the **seed variance halved** (0.76 → 0.32) — the
  anti-churn/horizon changes made the result far more consistent, which matters more for a rigor
  paper than the mean alone.
- Ordering **coevolution > single > population** has now held across all three runs — the robust part.
- Coevolution is the only RL arm beating Buy & Hold, with roughly **one-third its drawdown**.
- **Per-regime (V3):** bear — coevolution −1.72 vs B&H −4.72, single −3.46, population −4.64;
  sideways — coevolution **+0.29** (first time positive) vs B&H +0.42.
- **Still not statistically significant.** P(≤ B&H) = 0.131 is the closest yet but above 0.05.
  Coevolution also edges MA-crossover on mean Sharpe (−0.38 vs −0.47), but P(≤ MA) = 0.485 — a
  coin flip, so **do not claim it beats MA-crossover**. The next lever is **more test data**
  (walk-forward across windows, incl. a bull one), not more seeds.
- Everything lost money (down/choppy test period); comparison is risk-adjusted & relative.
  MA-crossover's respectable Sharpe is mostly from sitting in cash (defensive by construction).

**Arms-race diagnostics — the honest, persistent limitation:**
Disengagement was **not** cured, in any run. Environment-population fitness collapses in all three,
and `env-fresh-frac 0.25` did not prevent it:

| | V1 | V2 | V3 |
|---|---|---|---|
| Env fitness first → last | 2.78 → 0.61 | 2.87 → 0.42 | 2.19 → **0.21** |
| corr(internal, external) | +0.10 | −0.10 | +0.05 |

An early single-seed check during V2 *looked* fixed; averaged over 6 seeds it was not — verify
arms-race health across seeds, never from one run.

**The key finding is a decoupling of two questions.** V3's gains came from the **trader-side** fixes
(anti-churn + longer horizon), not from re-arming the adversary. So coevolution's out-of-regime
advantage **keeps improving despite an arms race that still dies out** — plausibly the adversarial
*early curriculum* is what helps, not the sustained late-stage race. "Does coevolution generalize
better?" (yes, consistently) is separate from "is our arms race well-behaved?" (no) — report both.
Fixing the arms race remains open work (bounded perturbations, asymmetric pacing, adaptive variation).

**Behavioral diagnosis that motivated the V3 fixes** (run on the V2 models):
the trader had learned a high-churn, cash-hugging policy — 116–146 trades bleeding 17–22% of capital
to fees/slippage, in-market only 2–17% of the time, and **4 of 6 seeds lost money on a rising (bull)
validation window**. A controlled A/B (same seed, fixes off vs on) cut trades 240 → 21, raised
time-in-market 31% → 85%, and turned −34.7% into **+7.8%** on that bull window. Note the residual
tension: the fixed agent is now close to buy-and-hold (≈85% in market), so it profits in uptrends but
does not reliably *exit* downtrends — being long-only, its only defense is moving to cash.

---

<a name="10-next-steps"></a>
## 10. Next steps

1. **Walk-forward across multiple test windows (including a bull one)** — now the single biggest
   lever. With P(≤ B&H) at 0.131, the binding constraint is only **9 test segments**, not seed
   count (V3 already has 6 seeds and a tight ±0.32 spread). `run_walkforward.py` is built and
   smoke-tested; it needs a compute slot.
2. **Fix the arms race** — disengagement survived both tuned variation rates and fresh blood.
   Remaining options: bounded perturbations of real segments (anticipated in `context.md` as the
   sanctioned extension), stronger asymmetric pacing (fewer trader timesteps per generation), or
   adaptive variation that scales up when adversary fitness drops.
3. **Decide on shorting.** The test window is bear/sideways and the env is long-only, so no agent
   can be *profitable* there — only less bad. Adding a short position raises the ceiling but changes
   the problem definition, so run it as a separate, clearly-labeled experiment.
4. **Tune the churn/participation balance** — V3's agent sits ≈85% in market (near buy-and-hold).
   Lowering `--trade-penalty` toward 0.001 should recover some selectivity (exit downtrends) without
   returning to the V2 churn.
5. Optional: **fitness-metric mini-ablation** (Sharpe vs Sortino) as an extra experiment.
6. Prospective/paper-trading validation on data frozen now, evaluated in a few weeks.

---
*Notes generated 2026-07-22; updated 2026-07-27 with the V3 run and the behavioral diagnosis.*
