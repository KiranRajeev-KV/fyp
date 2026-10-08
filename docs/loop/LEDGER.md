# Research loop ledger (brief: docs/loop/PROMPT.md)

Branch `loop/experiments` (from `fix/v5-audit` @ ebffc75). Deadline 2026-10-09 18:00. MAX_ITERATIONS 2.

## Pre-registered criteria
Section 4 of PROMPT.md, verbatim, frozen before the first arm run. Not edited.

## Benchmarks
### SYNTH_V1 (Task B, synthetic; separate from real data — never compared across benchmarks)
- Code: `ppo/src/synth_market.py`, sha256[:16] **8070ed6ffe695667** (also in `ppo/data/synth_v1/params.json`).
- Frozen 2026-10-08 ~22:56, after the cheap check passed and before any arm ran (R10).
- Parameters: semi-Markov regimes, geometric durations mean 480 bars; hourly drift bull +0.0004 / sideways 0 / bear −0.0008,
  vol 0.006 / 0.006 / 0.009; observed `regime_signal` = EMA(span 48) of (code ∈ {+1,0,−1} + N(0, 2.0²)); the 19 real features
  via `build_features.add_features`; StandardScaler fit on train.
- Mixes: train/val/control bull .47 / sideways .43 / bear .10; shifted bull .25 / sideways .25 / bear .50.
- Data `ppo/data/synth_v1/` (gitignored, rebuildable with `python synth_market.py --out ../data/synth_v1`):
  train 64,353 bars (pool 379 segments, stride 168: bull 45% / sideways 42% / **bear 12.9%**), val 6,552,
  dev_shift seeds 1000-1099 (200 segments, **58% bear**), dev_control seeds 2000-2099 (200 segments, 10.5% bear).
- Sealed: seeds 9000-9099 (`--sealed`), NOT built yet.

## Trials
| # | bench | what | data | outcome |
|---|---|---|---|---|
| 1 | synth (pre-freeze) | generator draft, train mix bear .12 | dev_shift/control, logistic probe | AUC 0.601 shift / 0.584 control; timer +0.76 vs B&H −3.57 (shift). Train series 19% bear: out of the 10-15% spec → retune train mix only |
| 2 | SYNTH_V1 | generator, train mix bear .10 → **frozen** | same | AUC 0.601 / 0.587 (signal alone 0.604); shift: timer **+0.69 vs B&H −3.57**, wins 153/200, timing +0.57; control: timer +2.37 vs B&H +1.65. PASS |

## Iterations
### iter01 (2026-10-08 22:50 –) Task B main run
- Code (all behind flags, defaults reproduce old behaviour): `segments.build_segment_pool` uses a `true_regime` column if present;
  `--segment-sampling regime_balanced` (EngineConfig.segment_sampling) for the `reweighted` arm (= population + PBT, oracle
  regime-balanced draws); `run_coevolution.py --features`; `run_v5.sh` TRAIN_DATA / VAL_DATA / FEATURES / EVAL + arm `reweighted`;
  `compare_arms.py --jobs` + timing score (R5). Smoke test extended (true labels + balanced sampling + generator) — PASSED.
