# CC-PPO code

Competitive coevolution of PPO trading agents for BTC. A population of PPO
traders is trained against a population of adversaries that choose which real
historical 30-day BTC windows the traders practise on. Evolution changes
hyperparameters only (population-based training); PPO's gradient does all
weight learning. See `../context.md` for the design.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
Python 3.12; a CPU is enough.

## Data

Data lives in `ppo/data/` (gitignored).

1. From the repo root: `python download_klines.py` (BTC spot 1h klines -> `ppo/data/raw/`).
2. Run the notebooks in `jupyter_project/` once (`data_collect` ->
   `feature_engineering` -> `train_test`). They write the base feature files
   and the feature scaler (`models/feature_scaler.pkl`).
3. From `ppo/src/`:
   ```bash
   python build_features.py --suffix _ext2         # extend features to the latest month
   python download_coin.py BTCUSDT --end 2026-08   # funding + open interest / long-short metrics
   ```
The six test periods (Mar 2022 – May 2026) are fixed in
`runs/walkforward/folds.json`. Bars from 2026-05-01 07:00 on are a sealed
holdout and are never used for development.

## Final pipeline

```bash
cd src
python coevo_eval.py                                     # train + score all arms on BTC
python coevo_seedpool.py r18a-progress-6-seeds r19a-progress-seeds48-53 \
    r21a-pure-alp r22a-pure-alp-seeds48-53              # the headline: 4 runs x 6 seeds
python coevo_floor.py r21a-pure-alp                      # exposure-floor variant (f = 0 … 0.75)
```
- `coevo_node.py` — the run's settings: generations, population sizes,
  features, adversary type. Set to the final configuration (learning-progress
  adversary). The headline pools four runs of it: seeds 42–47 and 48–53, each
  with and without the input-drift option noted in the file.
- `coevo_eval.py` — the fixed evaluation contract: cost model (agent = perp
  maker 0.02% fee + 0.01% slippage + funding; buy & hold = spot 0.1% + 0.05%),
  folds, test segments, selection and pass rules. Trains single / population /
  coevolution on each fold and scores them against buy & hold and MA crossover.
- Results go to `runs/coevo7/<run name>/`.

## Layout

```text
src/
├── coevo/
│   ├── engine.py             # generation loop, matches, internal vs external ledgers
│   ├── trader_population.py  # PPO traders + PBT (exploit / explore of hyperparameters)
│   ├── env_population.py     # adversaries: segment selection, fitness, GA
│   └── hall_of_fame.py       # archives for both populations (anti-cycling)
├── trading_env.py      # Gymnasium env: fees, slippage, funding, decision interval
├── segments.py         # 30-day segment pool, descriptors, walk-forward folds
├── regimes.py          # bull / bear / sideways labels (trailing, no lookahead)
├── baselines.py        # buy & hold, MA crossover (same cost model)
├── metrics.py          # Sharpe, Sortino, Calmar, max drawdown
├── build_features.py   # data pipeline (reproduces the notebooks)
├── coin_data.py        # bars + order flow + positioning + funding for one symbol
├── coevo_node.py       # run settings (final configuration)
├── coevo_eval.py       # fixed evaluation contract (see above)
├── coevo_seedpool.py   # pool reruns of one config that differ only in seeds
├── coevo_floor.py      # exposure floor: f * buy & hold + (1 - f) * agent
├── engine_cli.py       # shared EngineConfig command-line options
├── run_coevolution.py  # earlier entry point: three-way ablation on fixed splits
├── run_walkforward.py  # earlier entry point: rolling-window ablation
├── evaluate.py         # per-regime evaluation + bootstrap significance
└── compare_arms.py     # arm-vs-arm paired bootstrap tests
tests/smoke_test.py     # end-to-end run on synthetic data, no real data needed
jupyter_project/        # the original exploratory notebooks
run_*.sh                # launch scripts for the earlier multi-seed runs
```

Side experiments (kept so earlier results can be reproduced, not needed for
the final pipeline): `signal_probe.py`, `signal_hunt.py`, `carry_check.py`,
`order_flow.py`, `vol_target.py` (is there a timing signal?),
`synth_market.py`, `synth_market_v2.py`, `synth_dose.py` (synthetic
benchmark), `orx_*.py` (maker-fee strategy search), `train_ppo.py`,
`run_overnight.py` (early single-agent PPO and tuning).

## Smoke test

Runs the whole machinery (regimes, segment pool, env, baselines, all three
engine modes, halls of fame) on synthetic data in a few minutes:
```bash
cd tests && python smoke_test.py
```

## Trading environment

- **Observation:** scaled technical indicators (returns, EMAs, RSI, MACD, ATR,
  volatility, volume, Bollinger), optional order-flow / positioning / forecast
  features, and the account state.
- **Action:** target position (cash or long), taken every `decision_every`
  bars (4 in the final configuration).
- **Reward:** log return of portfolio value per step, after fees, slippage and
  funding.
