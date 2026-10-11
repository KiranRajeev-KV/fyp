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
### 9b. September 2026 audit (read before quoting any V4 number)

Full write-up: [`audit-2026-09.md`](audit-2026-09.md). The short version:

- **V4's per-generation ledgers were destroyed** by a relaunch on 2026-09-09 (16 of 18 runs);
  the frozen models and eval JSONs survive, so the headline table reproduces, but the curriculum /
  drift / cost analyses have no data behind them until V5 finishes. What survives is in
  `docs/artifacts/v4_evidence.tar.gz`.
- **The trader hall of fame was aliasing live weights and overwrote trader 0 every generation** in
  the coevolution arm only (an unintended between-arm difference). Fixed; V5 is the re-run.
- **Coevolution vs single / population was never significance-tested.** `compare_arms.py` now does
  it: on V4, P(coevolution ≤ single) = 0.17 and P(≤ population) = 0.26 (hierarchical bootstrap);
  coevolution wins 5/6 seeds on both.
- **Two more held-out months (Jul–Aug 2026, both rallies) erase coevolution's mean advantage**:
  on 11 segments coevolution −0.40, population −0.18, single −0.40 (`test_ext.parquet`). The bear
  advantage is unchanged; the cost in rallies is now measured. Only the walk-forward can say
  which dominates.
- **Disengagement has a structural cause:** adversary fitness = −(in-sample Sharpe) is capped at 0
  once the trader learns to hold cash, so no variation knob could ever fix it. V5 adds a
  `coevolution_regret` arm whose adversary is scored on regret vs the best trivial policy.
- Baselines now pay slippage; every final trader is saved; `config.json` records the git hash;
  the engine refuses to overwrite a finished run.

### 9c. V5 and walk-forward (2026-09-14/15) — the result that changes the framing

**V5** (`runs/v5/`, V4 config on the fixed engine, four arms incl. `coevolution_regret`, 6 seeds × 30 gens):

| 9-segment test window (rank-0 model) | Sharpe | bear | bull | sideways |
|---|---|---|---|---|
| coevolution | −0.16 ± 0.65 | −0.60 | +0.56 | −0.04 |
| single | −0.80 ± 0.60 | −1.96 | +2.34 | −0.73 |
| population | −1.43 ± 0.15 | −4.84 | +4.64 | −0.60 |
| coevolution (regret adversary) | −1.69 ± 0.32 | −4.41 | +4.25 | −1.24 |
| Buy & Hold / MA (MA now pays slippage) | −1.30 / −0.76 | | | |

Coevolution > population on 6/6 seeds (P = 0.11 hierarchical), > B&H 6/6 (P = 0.14), > single 4/6 (P = 0.22).
On the 11-segment window (adds the Jul–Aug 2026 rallies) coevolution −0.06 ≈ single −0.13 ≈ B&H −0.19.
The **regret adversary keeps the arms race alive in all 6 seeds** (adversary fitness 0.85–1.53 in the last 5
generations vs ≈0 under the original objective; zero identical policy pairs; value-head explained variance
+0.36..+0.56 vs V4's −0.44) **and produces the worst RL arm**: its traders behave like buy-and-hold. A sustained
arms race is not what buys bear robustness; the early bear-heavy curriculum of the collapsing adversary is.
Run-to-run noise is ≈0.5 Sharpe: the single arm's code did not change between V4 and V5 yet its mean moved
−1.32 → −0.80 (the training env seeds from OS entropy).

**Walk-forward** (`runs/walkforward/`, 6 rolling folds, test windows Mar 2022 → Apr 2026, 48 non-overlapping
segments: 8 bear / 11 bull / 29 sideways by trailing label; 3 seeds for the population arms, 6 for single,
20 generations, V5 engine flags). Per-seed final-population mean:

| arm | Sharpe | bear | bull | sideways |
|---|---|---|---|---|
| Buy & Hold | **+0.97** | −2.67 | +3.28 | +1.09 |
| population | +0.60 ± 0.09 | −2.71 | +2.59 | +0.75 |
| single | +0.41 ± 0.44 | −2.29 | +2.00 | +0.55 |
| coevolution | **+0.05 ± 0.07** | **−0.59** | +0.54 | +0.04 |
| MA-crossover | −0.68 | −2.44 | +0.91 | −0.79 |

Coevolution loses to population on 3/3 seeds (P = 0.89) and to single on 2/3 (P = 0.69); it beats only
MA-crossover. It is the best arm in the two bear-heavy folds (2022, 2025–26) and the worst in the four
bull/sideways folds.

**Reading.** The hypothesis "coevolution generalises better out of regime" does **not** hold on average. What
holds everywhere (V4, V5, 11 segments, all six folds) is narrower: coevolution yields a low-exposure trader that
loses far less in bear markets and gives up most of the upside elsewhere. The 9-segment headline was a
bear-and-sideways-only window that made that trade-off look like a general advantage. This is the rigorous
mixed result the project brief allows for; the methodological contribution (diagnostics, the regret control,
the walk-forward, the audit) is intact.

### 9d. V6 mixed curriculum (2026-10-08) — the abstention fix that did not fix it

The walk-forward failure had one mechanism: the coevolution trader abstains (1% exposure in every regime), so
its low drawdown is the drawdown of cash and its upside is zero. Two literature-motivated changes were run as a
fifth arm, `coevolution_mixed` (`runs/v6/`, 6 seeds × 30 gens, otherwise the V5 config):

* **Coverage mixing** (`--env-adversary-frac 0.5`, after Robust PLR): half of each match's segments come from
  the adversary, half from a uniform draw over the pool, and the adversary is scored **only on its own half**
  (`engine._match_segments`, `matches.csv:n_adversary_segments`). The trader therefore keeps seeing the whole
  distribution while the adversary still gets a clean signal.
* **Exploration floor + no churn tax** (`--ent-coef-min 0.005 --trade-penalty 0`): PBT can no longer evolve
  `ent_coef` down to the point where the policy settles on "hold", and holding is no longer the cheapest action.

**It moved the mechanism barely and the result the wrong way.**

| per-seed final-population mean | 9-seg test | 11-seg test | test exposure | train no-trade (last gen) |
|---|---|---|---|---|
| coevolution (V5) | −0.39 | −0.22 | 0.18 | 0.60 |
| coevolution_mixed (V6) | −0.73 | −0.52 | 0.21 | **0.24** |
| single | −0.80 | −0.13 | 0.42 | 0.00 |
| population | −1.44 | −0.34 | 0.96 | 0.00 |

Training-time abstention more than halved (no-trade fraction 0.60 → 0.24) but **test exposure rose only 0.18 →
0.21**, while trades per segment went 4.3 → 6.0 with no trade penalty to pay for them. Mixed loses to plain
coevolution on both windows (P = 0.77 hierarchical, 3/6 seeds) and to `single` on the 11-segment window
(P = 0.77, 2/6); on the 9-segment window it ties `single` (P = 0.48) and beats `population` (P = 0.18). The
arms race still decays (top adversary fitness 3.39 → 1.17 over 30 generations). It was therefore **not promoted
to the walk-forward**.

**Reading.** Abstention is not caused by the trade penalty, by an entropy collapse, or by the adversary starving
the trader of coverage — all three were removed and exposure did not return. It is caused by what Sharpe rewards
on these windows: with a long-only env and no short side, flat is the optimal response to any window the trader
cannot read, and the only lever that changes that is the action space or the objective — both of which are
frozen by the project framing. This is the third named failure mode, after disengagement (9b) and the
regret arm's buy-and-hold collapse (9c).

### 9e. No exploitable timing signal (2026-10-08) — why every method converges to a constant exposure

**Claim.** On real BTC hourly data the features carry almost no information about direction, so no
training method in this project can learn to time the market. Each arm instead settles on a **constant
exposure** fixed by its training distribution, and coevolution's adversary sets that constant low. The
curriculum changes in 9c-9d only move this exposure dial. Reproduce with
`python ppo/src/signal_probe.py [--funding ../data/funding/BTCUSDT_funding.parquet]` (6 walk-forward folds,
48 test segments of 720 bars, scaler and classifier fit on each fold's train span only, the last H train labels
purged). Output: `ppo/runs/walkforward/signal_probe.json`.

**1. The features barely predict direction.** Logistic regression (C = 0.1), label = sign of the forward
H-bar log return, mean of the per-fold test AUCs:

| features | AUC @ 24h | AUC @ 168h |
|---|---|---|
| the 19 engine features | **0.529** | 0.516 (pooled 0.48) |
| + 30-day and 90-day trend | 0.525 | 0.499 |
| + Binance funding rate | 0.526 | 0.512 |
| funding rate alone | 0.519 | 0.554 (pooled 0.48: not stable across folds) |

**2. Even ideal use of that signal does not beat buy-and-hold.** Ceilings on the same 48 segments, decisions
rebalanced daily, 0.15% cost per unit of position change, mean per-segment Sharpe:

| rule | Sharpe | wins vs B&H (of 48) | timing |
|---|---|---|---|
| buy-and-hold | **+0.93** (+0.97 with no entry cost) | — | 0 |
| logistic long/flat timer | −0.02 (+0.01 with no entry cost) | 15 | −0.06 |
| logistic long/short | −1.12 | 16 | −0.12 |
| volatility targeting (train-median 7d vol / trailing 7d vol, cap 1) | +0.77 | 9 | +0.03 |
| 30-day trend filter (close > 720-bar SMA) | −0.10 | 13 | +0.39 |

The trend filter does move exposure with the segment's direction (timing +0.39), but only after the move has
happened, and the whipsaw costs more than it saves.

**3. The trained arms do not time either.** Timing score (R5) = exposure on segments where buy-and-hold
gained minus exposure where it lost; corr pooled over every (seed, segment) cell. Walk-forward, `best_trader`:

| arm | exposure up / down | timing | corr(exposure, segment move) |
|---|---|---|---|
| coevolution | 0.12 / 0.18 | −0.05 | −0.05 |
| single | 0.57 / 0.61 | −0.05 | −0.10 |
| population | 0.85 / 0.89 | −0.04 | −0.21 |

All three timing scores are slightly negative, so exposure is a constant. Ranking the arms by exposure
(population > single > coevolution) also ranks them by walk-forward Sharpe (+0.60 > +0.41 > +0.05; B&H +0.97) in a
market that went up.

**4. What this rules out.**
* *A better reward.* An excess-over-buy-and-hold per-step reward has the same optimal policy as the log-return
  reward, because the buy-and-hold term does not depend on the action. It can reduce variance, but it cannot
  create a signal.
* *A better selector.* Within a trader, the match-to-match sd of fitness (1.17) is larger than the between-trader
  fitness sd (0.82), and internal rank vs held-out Sharpe has Spearman 0.02. PBT and the hall of fame are
  selecting on noise (see also `hyperparam-search-is-drift`). Run-to-run noise is ~0.5 Sharpe.
* *More curriculum tuning.* The adversary can only change which windows the trader sees, and therefore the
  exposure level the trader converges to. That helps in bear-heavy folds and hurts in bull-heavy ones, which is
  exactly the walk-forward pattern in 9c.

**Reproduction status** (Task A acceptance, ±0.01 against the original ad-hoc numbers): AUC@24h, funding,
trend features, buy-and-hold (both cost conventions), long/flat timer (+0.013 vs +0.01, no entry cost),
trend filter (−0.103 vs −0.10) and all six arm exposures and correlations reproduce. The volatility-targeting
Sharpe is +0.771 against +0.76 (just outside ±0.01; same 9/48 wins). Long/short (−1.12, or −1.08 with no entry
cost, against −0.97) and AUC@168h (0.516 against ~0.53) do **not** reproduce, because the original ad-hoc code was
not kept. Neither changes the conclusion. The trend filter turned out to be "close above its 30-day SMA"
(30-day momentum > 0 gives +0.23).

**What follows.** The test this project can still run is on a market where timing *is* possible. The
controlled synthetic benchmark (Task B, `ppo/src/synth_market.py`, `docs/loop/LEDGER.md`) provides that, with a
known regime and a deliberate train/test regime shift.

### 9f. Synthetic regime benchmark (2026-10-08..10) — where timing is possible, coevolution wins under shift

**Why a synthetic market.** §9e shows that real BTC has no usable direction signal. On real BTC every arm therefore reduces to an exposure dial, and the question the project asks cannot be tested there: does coevolution generalise better under a regime shift when there *is* something to learn? A generated market isolates that question. In it the regime is known, it is partly observable, and its mix is deliberately shifted between train and test. This is a controlled test of the training method, not evidence about BTC. The two benchmarks are never compared with each other.

**How.**
- SYNTH_V1 (`ppo/src/synth_market.py`, sha256[:16] `8070ed6ffe695667`, frozen before any arm ran):
  - semi-Markov bull / sideways / bear regimes, mean duration 480 bars;
  - hourly drift +0.0004 / 0 / −0.0008, vol 0.006 / 0.006 / 0.009;
  - a noisy observed `regime_signal`, plus the 19 engine features from `build_features.add_features`.
- Mixes: train and control sets are 10% bear; the shifted test sets are 50% bear.
- Logistic probe AUC 0.60, against 0.53 on real BTC.
- SYNTH_V2 (`ppo/src/synth_market_v2.py`, `bd5b061d7886a3ae`) is the weaker-predictability market, AUC 0.55.
  - It raises signal noise 2 → 12 and multiplies all drifts by 0.6.
  - This deviates from the pre-registered "change only the signal". Noise alone cannot lower the AUC, because the price features carry the regime. The user approved the change before the freeze.

**Protocol.**
- Criteria were frozen in `docs/loop/PROMPT.md` §4. Every trial was written into `docs/loop/LEDGER.md` before it ran.
- Five arms: single, population, coevolution, coevolution_regret, and the oracle `reweighted`. The oracle is population + PBT trained on regime-balanced segments, so it uses the true labels that coevolution never sees.
- 6 seeds × 20 generations, V5 config.
- Score: final-population mean Sharpe per seed. Significance: hierarchical bootstrap P(coevolution ≤ X).
- Dev sets were used during development. Each sealed set (new generator seeds) was scored once.

| shifted test (≈50-58% bear) | V1 dev | V1 sealed | V2 dev | V2 sealed |
|---|---|---|---|---|
| coevolution | **+0.60** | **+0.90** | **−0.20** | **+0.02** |
| reweighted (oracle, uses labels) | +0.45 | +0.78 | −0.53 | −0.13 |
| population | −0.00 | +0.41 | −0.72 | −0.35 |
| single | +0.10 | +0.36 | −0.91 | −0.48 |
| buy & hold | −3.6 | −2.13 | −2.05 | −1.25 |
| P(coevo ≤ single / population) | 0.003 / 0.001 | 0.001 / 0.004 | 0.003 / 0.008 | 0.010 / 0.041 |
| P(coevo ≤ reweighted) | 0.11 | 0.24 | 0.035 | 0.17 |

- **Claim tier MODERATE on both markets, sealed-confirmed.**
  - Under a bear-heavy shift, coevolution beats single and population.
  - It at least ties the oracle without seeing regime labels.
  - `coevolution_regret` fails the criteria on both markets.
- **Dose-response** (V1, the trained models, 500 fresh segments, `ppo/figures/synth_dose_response.png`):
  - Coevolution minus single grows +0.12 → +0.23 → +0.34 → +0.51 → +0.71 as the bear share goes 13 → 25 → 40 → 58 → 75% of segments.
  - It is significant from 40% up. The sign never changes in the tested range.
- **The trade-off.** On the unshifted control set coevolution is significantly *worse*: −0.26 (V1) and −0.39 (V2) vs single. The oracle is the best arm there.
- **Mechanism.**
  - All arms learn the timing; exposure separates bull from bear.
  - Coevolution differs by holding a lower exposure level: 0.23-0.33, against 0.41-0.70 for the others.
  - That level is set by the adversary's bear-heavy curriculum (50-56% bear served at generations 5-10, from a 12.9%-bear pool).
  - It pays off when the test is bear-heavy and costs when it is not. This is the same exposure dial as on real BTC, but here it sits on top of real timing.

### 9g. Shared exam (`--shared-exam 16`) — the best variant, synthetic only

**Why.** Selection is a lottery (§9e, B2 in the audit). One trader's match-to-match sd (1.17) is larger than the spread between traders (0.82).

**Change** (`EngineConfig.shared_exam`, `engine._shared_exam`):
- Each generation, every trader sits the *same* 16 uniform segments, and PBT, `best_trader` and the hall of fame rank on that score.
- Each adversary is credited with the mean over *all* traders on the segments it proposed.

**Results.**
- Dev (trial 18): scored on V1 dev only.
- Fresh sealed (trial 19): new seeds 9500-9699 (`ppo/data/synth_v1_sealed2/`), pre-registered, scored once (`ppo/runs/loop/iter03_exam/`).

| final-population mean | exam | coevolution | reweighted | population | single | B&H |
|---|---|---|---|---|---|---|
| dev shift | **+0.70** | +0.60 | +0.45 | −0.00 | +0.10 | −3.6 |
| dev control | +2.38 | +2.13 | **+2.52** | +2.42 | +2.40 | |
| sealed2 shift (50.6% bear) | **+1.11** (sd 0.08) | +0.95 | +0.91 | +0.46 | +0.52 | −3.00 |
| sealed2 control (12.7% bear) | **+2.71** (sd 0.08) | +2.45 | +2.60 | +2.51 | +2.51 | +1.74 |

- **Confirmed on the fresh sealed sets:**
  - It beats single and population under shift (P 0.011 / 0.000).
  - It beats the label oracle under shift (+0.20, P 0.048, 5/6 seeds).
  - It is the best arm on control (P 0.08 / 0.04 vs single / population), so the control cost of plain coevolution is gone.
  - Its seed sd is about 3× smaller.
- **Not confirmed:**
  - Better than plain coevolution under shift (+0.15, P 0.16).
  - Better rank-0 selection, which was the primary pre-registered measure: the best-minus-final gap is unchanged at −0.001.

### 9h. Last attempts on real BTC (2026-10-10..11) — nothing beats buy & hold

Each attempt was pre-registered in the ledger and used the same 48 walk-forward segments (2022-03 → 2026-05). Each attempt's pass gate is in its ledger entry. The core requirement throughout: mean Sharpe above B&H (+0.93) with paired-bootstrap P ≤ 0.10. The carry and vol-target checks also had to stay above B&H at 2× costs.

| attempt (trials) | best result | vs B&H +0.93 | why it fails |
|---|---|---|---|
| Signal hunt: S&P / VIX / DXY / 10y, on-chain (CoinMetrics), weekly momentum; logistic and rule timers, 24h and 168h (S1-S12, `signal_hunt.py`) | best AUC 0.52; best timer +0.94 | P 0.47 | no set reaches AUC 0.55. The near-tie is 79% always-long (timing ≈ 0). Momentum rules time (+0.26-0.34) but whipsaw. |
| Shared-exam arm on the walk-forward, 3 seeds × 6 folds (32, `runs/walkforward_exam/`) | −0.03, exposure 0.23 | P 0.99 (vs +0.97, compare_arms convention: no entry cost) | another exposure setting, timing −0.12. Arms again rank by exposure: population +0.60 > single +0.41 > coevolution +0.05. |
| Funding carry: long spot + short perp (33-35, `carry_check.py`) | always-carry **+2.04** | P 0.032 | **fails at 2× cost (+0.16)**. Earns +0.30% per month vs B&H +2.39%. Switching on the funding sign whipsaws (−8.6, +1.25). |
| Volatility targeting: min(1, train-median σ / trailing σ), daily (36-37, `vol_target.py`) | +0.77 (7-day σ), +0.64 (1-day σ) | P 1.00 | fully invested about 95% of the time. Cuts after vol spikes miss the rebounds. |

- The 7-day volatility-targeting rule had already appeared in §9e as a ceiling (+0.77).
- Trials 36-37 re-ran it under a formal pass gate, added a 1-day window and the 2× cost check, and reproduced it exactly (+0.771).
- **Trial count: 37 in total, 19 of them on real BTC** (4, S1-S12, 32-37).
- Loosening any gate after the fact, e.g. dropping carry's 2× cost check, would be post-hoc selection, so it was not done.

## 10. Conclusions and what remains

**What the project shows.**
1. **Real BTC, 2022-2026: no method in this project and no simple rule beats buy & hold robustly.**
   - The features, plus cross-asset, on-chain, momentum and funding data, do not predict direction (AUC ≤ 0.53).
   - Every RL arm converges to a constant exposure, and arms rank by that exposure in a rising market.
   - Coevolution's adversary sets a low exposure. That makes it the best arm in bear folds and the worst overall (§9c, §9e, §9h).
2. **Where a learnable regime signal exists, coevolution generalises better under a bear-heavy shift.**
   - Shown on two synthetic markets, sealed-confirmed, with a monotone dose-response (§9f).
   - The shared-exam variant beats the label oracle and removes the cost on unshifted data (§9g).
   - The mechanism is an exposure level matched to the shift. That is why it cannot help where there is no signal.
3. **Methodological contributions:**
   - the audit (hall-of-fame aliasing bug, lost V4 ledgers, §9b);
   - hierarchical arm-vs-arm tests;
   - the walk-forward;
   - the regret-adversary control, which shows that a sustained arms race is not what helps (§9c);
   - the pre-registered ledger with a trial count.

**Panel-safe one-liner.**
- "On real Bitcoin nothing we tried, RL or rule, reliably beats buy-and-hold; we show why (no direction signal).
- On a controlled market where a signal exists, coevolution beats standard PPO and population training under regime shift, and its shared-exam variant beats a label-using oracle."

**Remaining work.**
- Writing:
  - a one-page panel Q&A;
  - the report chapters (results and discussion from §9c-§10);
  - slides 8-13 last;
  - merging `loop/experiments` into `main`.
- Not done, and outside the current framing (flag before starting):
  - a short side or a different action space;
  - a non-Sharpe objective;
  - other synthetic generators.
- Prospective check: the real bars from 2026-09-01 are still sealed and untouched. They could be scored once on the frozen models as a final out-of-sample look.

---
*Notes generated 2026-07-22; updated 2026-07-27 (V3), 2026-09/10 (§9b-9e), 2026-10-12 (§9f-9h, §10 rewritten).*
