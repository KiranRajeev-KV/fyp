# CC-PPO code

Competitive coevolution of PPO trading agents for BTC. Traders learn through
PPO and evolve three hyperparameters through population-based training (PBT).
The environment population selects historical windows. The final BTC node
uses absolute learning progress, shared exams, and population voting; earlier
scripts retain other objectives. See [the current design](../context.md#current-implementation-final-btc-configuration)
and [the dated experiment record](../docs/experiment-notes.md).

## Setup

All shell examples in this README start from the **repository root** and use
an activated environment. Use Python 3.12; a CPU is enough.

```bash
python3.12 -m venv ppo/.venv
source ppo/.venv/bin/activate
python -m pip install -r ppo/requirements-cpu.txt
python -m pip check
```

| File | Purpose |
|---|---|
| `requirements.txt` | Pinned direct runtime dependencies, including download and plotting tools |
| `requirements-cpu.txt` | Runtime pins plus the matching CPU-only PyTorch build |
| `requirements-notebooks.txt` | Runtime pins plus Jupyter, kernel support, and Excel export |

For CPU notebooks, install both sets and open the notebook directory:

```bash
python -m pip install -r ppo/requirements-cpu.txt -r ppo/requirements-notebooks.txt
jupyter lab ppo/jupyter_project
```

The notebooks use relative paths: their kernel working directory must be
`ppo/jupyter_project/`. For GPU use, install the runtime or notebook file
with the matching PyTorch build for your hardware. Direct dependencies are
pinned; transitive dependencies are not fully locked. The CPU baseline was
checked on Python 3.12.13 with the synthetic smoke test.

Python script defaults resolve relative to `ppo/`, independent of the caller's
working directory. Explicit CLI paths resolve relative to the caller's directory.
Shell launchers select their own directory. `coevo_seedpool.py` and
`coevo_floor.py` run through guarded `main()` functions, so importing them
does not read run files or launch workers.

## Data and evaluation windows

Data, scalers, and models are generated files under `ppo/data/` and `ppo/models/`.
They are not included in a clean checkout.

1. Run `python download_klines.py` to download spot hourly bars into
   `ppo/data/raw/`. Its fixed experiment cutoff is August 2026.
2. Run `data_collect.ipynb`, `feature_engineering.ipynb`, and `train_test.ipynb`
   in order, with the notebook working directory described above. They create
   the cleaned/base feature files, processed splits, and fitted feature scaler.
3. Back at the repo root, extend/check the base features and download the
   coin-layout inputs through the same fixed August cutoff:

   ```bash
   python ppo/src/build_features.py --suffix _ext2
   python ppo/src/download_coin.py BTCUSDT --end 2026-08
   ```

`build_features.py` requires existing raw files, the base feature file, fitted
scaler, and output directories; it is an extension tool, not a complete
notebook replacement. Its numerical overlap check does not fully validate
missing timestamp matches or nonfinite values. The BTC loader also supports
an older local metrics/funding layout; see `coin_data._paths` and `_funding`
if using archived data instead of the coin downloader's directory layout.

The tracked `runs/walkforward/folds.json` defines six development folds,
covering 2022-03-23 05:00 through 2026-05-01 06:00 UTC. Each fold supplies eight
non-overlapping 720-bar scoring segments; residual bars are not scored. Test
windows follow each corresponding model's training window; earlier tests can
be included in later folds' training history.

The final BTC evaluator excludes bars from 2026-05-01 07:00 UTC onward from
training and scoring, although loaders may read the full data files. The
historical May–September 2026 holdout/prospective checks have been consumed.
Their spent status is recorded in the experiment notes; they are not fresh
future evidence. Those notes proposed October onward as the next window at
October 12, and later usage must be checked before claiming freshness.

## Final BTC run matrix

`src/coevo_node.py` contains one `CONFIG`. It currently selects the third row
below. To produce the headline, run each row separately: edit `CONFIG['name']`,
`CONFIG['seeds']`, and the coevolution arm's drift options, then execute
`python ppo/src/coevo_eval.py` before moving to the next row. Keep the other
settings unchanged. These are four configurations, not four commands that
are automatically expanded by the evaluator.

| Run name | Seeds | Input drift in coevolution arm |
|---|---|---|
| `r18a-progress-6-seeds` | 42–47 | Enabled |
| `r19a-progress-seeds48-53` | 48–53 | Enabled |
| `r21a-pure-alp` | 42–47 | Disabled |
| `r22a-pure-alp-seeds48-53` | 48–53 | Disabled |

For the enabled rows, add these keys to
`CONFIG['arm_overrides']['coevolution']`; remove them for the disabled rows:

```python
"obs_adversary": ["rhat4", "rhat24", "rhat72"],
"obs_bound": [0.75, 0.4],
"obs_sign_random": True,
```

Each run writes `ppo/runs/coevo7/<name>/BTCUSDT/`, including `contract.json`,
per-fold models and `final_traders/`, `per_segment.csv`, and `result.json`.
The four completed per-segment CSVs must exist before pooling:

```bash
python ppo/src/coevo_seedpool.py r18a-progress-6-seeds r19a-progress-seeds48-53 \
    r21a-pure-alp r22a-pure-alp-seeds48-53
python ppo/src/coevo_floor.py r21a-pure-alp
```

The floor evaluator additionally needs the named run's contract, final
populations, and source data. It writes a CSV for buy-and-hold exposure floors
0, 0.25, 0.5, and 0.75; `MULT=2` selects doubled costs.

The headline averages four run-level seed means with equal weight: 24
seed/configuration runs, using 12 distinct seed IDs. Pooling checks fold and
segment alignment, but does not verify configuration equivalence or weight
individual seeds across unequal-size runs. The four runs above each have six
seeds. Their generated `coevo7` artifacts are not tracked, so the reported
headline cannot be independently verified from this checkout alone.

## Final evaluation contract

- Node settings: 12 generations and six traders (one in the single arm).
  Coevolution uses 12 adversaries, a half-uniform training curriculum, an
  eight-window shared exam, and learning progress with regime lock. Other
  arms retain match-based selection fitness.
- Observations: `volatility_168h`, `ema50_ema200_ratio`, three ridge forecasts
  (`rhat4`, `rhat24`, `rhat72`), and three account values. Explicit selected
  inputs override the base list; forecast predictors still use base/flow data.
- Actions: target cash or long every four hourly bars; episodes start long.
- Scoring: `score='vote'` averages final traders' action probabilities and
  chooses the maximum, then averages scores across seeds. Other node values
  use the saved internally ranked `best_trader.zip`.
- Costs per side: agents pay 0.02% fee + 0.01% slippage and funding while long;
  spot buy-and-hold pays 0.1% + 0.05%; MA crossover uses the agent fee/slippage
  without funding.
- Significance: paired bootstrap over segment differences after averaging
  seeds; the historical `compare_arms.py` also has seed/hierarchical tests.
  The final evaluator does not implement a deflated-Sharpe selection correction.
- Pass rule: positive coevolution gaps against all four references, bootstrap
  probabilities ≤ 0.10, and a higher Sharpe than buy-and-hold at doubled costs.
  A reported mean advantage does not by itself satisfy this rule.

Training forecasts use purged block cross-fitting with other training blocks
on both sides of the held-out block. This is not strictly chronological
training prediction; external test forecasts fit purged training labels.
Learning-progress history is currently omitted from checkpoints, so resume
restarts that part of the adversary state.

## Layout and earlier experiments

| Location | Role |
|---|---|
| `src/coevo/` | Engine, trader PBT, adversary evolution, and archives |
| `src/trading_env.py` | Fee/slippage/funding environment; current-row features and account state |
| `src/feature_columns.py` | Shared ordered 19-feature base schema |
| `src/project_paths.py` | Paths relative to the project rather than the caller's directory |
| `src/build_features.py`, `coin_data.py` | Feature extension and BTC evaluation inputs |
| `src/coevo_node.py`, `coevo_eval.py` | Final BTC settings and evaluation |
| `src/coevo_seedpool.py`, `coevo_floor.py` | Run pooling and exposure-floor evaluation |
| `src/segments.py`, `regimes.py` | Segment descriptors, regime labels, and rolling folds |
| `src/baselines.py`, `metrics.py` | Baselines with caller-selected costs and performance metrics |
| `src/engine_cli.py`, `run_coevolution.py`, `run_walkforward.py` | Shared subset of engine flags for legacy entry points |
| `src/evaluate.py`, `compare_arms.py` | Legacy per-regime and arm-vs-arm evaluation |
| `tests/test_cli_imports.py` | Fast regression tests for import behavior and CLI wiring |
| `tests/test_synthetic_metadata.py` | Named benchmark metadata and dose-response build regression tests |
| `tests/smoke_test.py` | End-to-end synthetic engine check |
| `jupyter_project/` | Original notebooks; retain their own working-directory assumptions |
| `run_*.sh` | Legacy multi-seed launchers, not the final BTC run matrix |

The original notebook feature lists are historical copies; maintained Python
scripts share `feature_columns.py`. The V5 launcher defaults to 30 generations,
while the older walk-forward launcher defaults to 20, so their full configs
are not identical. `run_overnight.py` is the earlier V2 tuning pipeline.

Side experiments: signal diagnostics (`signal_probe.py`, `signal_hunt.py`,
`order_flow.py`, `vol_target.py`, `carry_check.py`), synthetic benchmarks
(`synth_market.py`, `synth_market_v2.py`, `synth_dose.py`), and historical ORX
strategy checks (`orx_*.py`). Task A/B/C refer respectively to signal diagnosis,
the synthetic benchmark, and shared exams. R-number labels identify entries
in the original local experiment ledger, which is not included here.

Synthetic metadata records benchmark names and parameters. Source-file hash
fields and checks have been removed; older local metadata may retain them.
The benchmark parameters and recorded historical result tables are unchanged.

## Checks

From the repo root, with the environment active:

```bash
python -m pip check
python -m unittest discover -s ppo/tests -p 'test_*.py'
python ppo/tests/smoke_test.py
```

The smoke test exercises the environment, segments, baselines, engine modes,
archives, checkpoint/resume plumbing, and shared exams using synthetic data.
It verifies execution, not trading performance or identical progress-state
continuation after a restart. No real data download is required.
