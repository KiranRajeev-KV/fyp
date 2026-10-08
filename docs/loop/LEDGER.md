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
| 3 | SYNTH_V1 | plumbing: `single` seed 99, 2 generations (compare_arms code test only, not used for any choice) | dev_shift | Sharpe −2.18 vs B&H −3.55; timing +0.42 (exposure bear .26 / bull .74). Even a 2-gen PPO times the regime here |
| 4 | REAL (re-run, not new variants) | Task A `signal_probe.py`: logistic AUC (19 feats / +trend / +funding / funding only) at 24h, 168h; ceilings long/flat, long/short, vol-target, 30d SMA filter, B&H | 6 walk-forward folds, 48 segs | AUC@24h 0.529; timer −0.02 (+0.013 no entry cost) vs B&H +0.93 (+0.97); L/S −1.12; vol-target +0.77 (9/48); SMA filter −0.10; arm timing all ≤ 0. See notes §9e |

## Iterations
### iter01 (2026-10-08 22:50 –) Task B main run
- Code (all behind flags, defaults reproduce old behaviour): `segments.build_segment_pool` uses a `true_regime` column if present;
  `--segment-sampling regime_balanced` (EngineConfig.segment_sampling) for the `reweighted` arm (= population + PBT, oracle
  regime-balanced draws); `run_coevolution.py --features`; `run_v5.sh` TRAIN_DATA / VAL_DATA / FEATURES / EVAL + arm `reweighted`;
  `compare_arms.py --jobs` + timing score (R5). Smoke test extended (true labels + balanced sampling + generator) — PASSED.
- Plumbing (2 gens, seed 99, all 5 arms, `runs/loop/iter01_plumbing/`): all ran clean, ~6.5 min per 2 gens for coevolution.
  Served regime mix (non-HOF matches): population bear 10%, **reweighted 34/33/33**, coevolution bear 25%, regret bear 23%.
- **Launched 23:05** `runs/loop/iter01_taskB/` (log `ppo/runs/loop/iter01.log`, per-run `ppo/runs/iter01_taskB_logs/`), git 2dc22eb:
  arms coevolution_regret, coevolution, reweighted, population, single × seeds 42-47, GENERATIONS=20, JOBS=6 THREADS=2,
  V5 config otherwise. Relaunch command (resumes):
  `cd ppo && D=../data/synth_v1 && GENERATIONS=20 JOBS=6 THREADS=2 EVAL=0 ARMS="coevolution_regret coevolution reweighted population single" OUTDIR=../runs/loop/iter01_taskB TRAIN_DATA=$D/train.parquet VAL_DATA=$D/val.parquet FEATURES=$D/feature_columns.pkl nohup bash run_v5.sh >> runs/loop/iter01.log 2>&1 &`
- Eval plan (after runs): `compare_arms.py --final-traders --jobs 12` on dev_shift and dev_control, arms in that order; criteria = §4.

### Task A (done 2026-10-08 23:40, while iter01 runs)
- `ppo/src/signal_probe.py` (+ `ppo/data/funding/BTCUSDT_funding.parquet`, built from the earlier session's Binance zips, 2020-01..2026-09).
- Acceptance (±0.01 vs PROMPT §1): PASS for AUC@24h 0.529, trend feats, funding (0.519 / adds nothing), B&H 0.927 & 0.968,
  long/flat timer +0.013 (no-entry-cost convention), trend filter −0.103 (= close > SMA720; momentum>0 gives +0.23), all arm
  exposures/corrs. MISS: vol-target +0.771 (vs 0.76, +0.011), long/short −1.12/−1.08 (vs −0.97), AUC@168h 0.516 (vs ~0.53).
  The originals' code is lost; conventions searched for ~15 min then stopped. No conclusion changes.
- `compare_arms.py` timing score: corr is now pooled over (seed, segment) cells = the §1 definition (reproduces −0.05/−0.10/−0.21).
- `docs/experiment-notes.md` §9e written.
- Real-data trial count so far: not reconstructed beyond this loop (V1-V6 + walk-forward + ad-hoc checks ≈ 10+ variants);
  no real-data claim is being made, so no DSR yet.
- 02:45 interim (training logs only, no test data touched): coevolution_regret done 00:57, coevolution done 02:31, 0 errors.
  Training exposure gen0→9→19: regret 0.62→0.77→0.75, coevolution 0.60→0.72→0.69 — **no collapse to cash** (fallback not triggered).
  Adversary served bear share (pool 12.9%): coevolution 0.24→0.50 (g5)→0.56 (g10)→0.27 (g19); regret 0.24→0.33→0.26→0.26.
