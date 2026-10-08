You are running a self-paced research loop on my final-year project, CC-PPO (competitive coevolution of PPO
BTC traders vs adversarial market-segment selectors), repo /home/nandu/uni/fyp/fyp, code in ppo/. Each time you
wake up, run ONE step of the protocol below, then schedule your next wake-up. All state lives on disk in
docs/loop/LEDGER.md (create it on the first iteration). Read it before doing anything else. Your memory index
(MEMORY.md) also has background notes: v6-mixed-curriculum, no-timing-signal, audit-sep-2026.

HARD DEADLINE: final summary in LEDGER.md by 2026-10-09 18:00 (loop started ~2026-10-08 23:00).
MAX_ITERATIONS: 2 (the main Task B experiment + at most one retry).

TIME BUDGET (plan to it; the machine has 12 cores / 14 GB, JOBS=6 THREADS=2; a 20-generation coevolution
run takes ~1.5 h, so all Task B arms x 6 seeds take ~6 h):
  by ~02:30  Task B generator built, cheap check passed, frozen; `reweighted` arm coded; smoke test and
             2-generation plumbing run pass; full Task B run LAUNCHED (GENERATIONS=20)
  while it runs  Task A (no compute needed)
  by ~10:00  Task B analysed against section 4; sealed holdout run once if criteria are met
  10:00-16:00  buffer for ONE retry (crash, or the generator failing in a way the cheap check missed);
             never a retry chosen because arm results were disappointing (R10)
  by 18:00   final summary written; loop stopped
If a step overruns by more than 1 h, cut scope (fewer arms is NOT allowed; drop Task A's write-up polish
first) rather than miss the deadline.  Branch: create `loop/experiments` from `fix/v5-audit` on
the first iteration; commit at the end of every iteration; never push.

CONSTRAINT DECISION: the controlled synthetic-market benchmark (Task B) is APPROVED as a separate, clearly labelled
experiment. The real-data arms keep all frozen constraints.

==================================================================================================
1. CONTEXT: WHERE THE PROJECT STANDS (verified 2026-10-08)
==================================================================================================
Arms: single (1 PPO trader, uniform windows), population (8 traders + PBT, uniform windows), coevolution
(8 traders + PBT vs 16 adversaries that pick training windows from 379 historical 720-bar BTC segments),
plus variants coevolution_regret (V5) and coevolution_mixed (V6). Env: long-only, Discrete(3) all-in/all-out,
fee 0.1% + slippage 0.05%; per-step reward = log return; fitness = per-segment Sharpe.

Results:
- V5 9-seg test (Oct 2025+): coevolution -0.16 > single -0.80 > population -1.43; looked like a win.
- Walk-forward (6 folds, Mar 2022 - Apr 2026, 48 segments) reversed it. Final-pop mean Sharpe: B&H +0.97 >
  population +0.60 > single +0.41 > coevolution +0.05. Coevolution wins only in the 2 bear-heavy folds.
- V5 regret adversary: keeps the arms race alive in 6/6 seeds but gives the WORST arm (buy-and-hold-like).
- V6 coevolution_mixed (half adversary / half uniform windows, ent_coef floor 0.005, no trade penalty):
  training no-trade fell 0.60 -> 0.24, but test exposure only 0.18 -> 0.21 and Sharpe got worse.
  Not promoted. Written up in docs/experiment-notes.md §9d.

Root cause, found by ad hoc checks (code NOT in the repo; rebuild it in Task A):
- No arm times the market. Exposure in up vs down walk-forward segments: coevolution 0.12 vs 0.18,
  single 0.57 vs 0.61, population 0.85 vs 0.89. corr(exposure, segment B&H return) = -0.05 / -0.10 / -0.21.
  Each arm learns a constant exposure level set by its training distribution.
- Features barely predict direction. Logistic regression (C=0.1, StandardScaler fit on the fold's train span,
  label = sign of forward H-bar log return, purge H bars before test) on train_ppo.FEATURE_COLUMNS over the 6
  walk-forward folds (runs/walkforward/folds.json): AUC 0.529 at 24h, ~0.53 at 168h. 30- and 90-day
  trend features don't help. Binance funding rate (data.binance.vision/data/futures/um/monthly/
  fundingRate/BTCUSDT/, from 2020-01, merge_asof backward on bar close time) AUC 0.51 and adds nothing.
- Ceiling checks on the 48 segments (daily rebalance, 0.15% cost per position change):
  logistic long/flat timer +0.01 vs B&H +0.97; long/short -0.97; volatility targeting (train-median 7d vol /
  trailing 7d vol) +0.76 vs B&H +0.93 (same cost model), wins 9/48; 30-day trend filter -0.10.
- An excess-over-B&H per-step reward has the SAME optimal policy as the current reward (the B&H term is
  action-independent), so it only reduces variance. Ruled out as a lever.
- Selection is a lottery: within-trader match sd 1.17 > between-trader fitness sd 0.82; rank vs held-out
  Spearman 0.02. Run-to-run noise is ~0.5 Sharpe.
Conclusion so far: on real BTC no training method can time the market, so curriculum changes only move an
exposure dial.

Tools: ppo/src/compare_arms.py (--arms name='glob', --data parquet | --walkforward-root, --final-traders;
segment / seed / hierarchical bootstraps), ppo/run_v5.sh (env: ARMS, SEEDS, GENERATIONS, OUTDIR, JOBS, THREADS;
arm flags are defined in its case block), ppo/run_walkforward.sh (same, plus FOLDS), ppo/src/engine_cli.py
(shared flags), ppo/tests/smoke_test.py (must pass before any launch). The engine refuses to overwrite a
populated out_dir and checkpoints every generation (--resume). Use .venv: ppo/.venv/bin/python.
The machine has died mid-run twice: if a run's log stops with no python process alive (pgrep -f
run_coevolution), relaunch the same command; finished runs are skipped and the rest resume.
JOBS=6 THREADS=2 is the proven parallel setting.

==================================================================================================
2. RULES (these protect the result from being dismissed by the panel; do not relax them)
==================================================================================================
R1 Pre-registered success. "Better" means meeting the criteria in section 4, not "a bigger number".
R2 Trial ledger. Every variant evaluated on any data (including the cheap checks) gets a numbered row in
   LEDGER.md. Report the total trial count with any claimed result, and for real data compute the
   Deflated Sharpe Ratio (Bailey & Lopez de Prado) and/or Probability of Backtest Overfitting (CSCV).
R3 Sealed holdouts, touched ONCE at the end: synthetic generator seeds 9000-9099; real BTC bars from
   2026-09-01 onward. Choose variants only on dev data.
R4 Check before running. Before any run over 1 h, do a check of under 15 min that could kill the idea
   (a ceiling check, a probe, or a 2-generation plumbing run) and log the outcome.
R5 Timing, not exposure. Every evaluation reports a timing score: exposure in up-segments minus
   exposure in down-segments, plus corr(exposure, segment move). A Sharpe gain with timing <= 0 is an
   exposure-dial artefact. Log it as such; it is not a success.
R6 Frozen for real-data arms: per-segment Sharpe fitness, long-only action space, historical segments.
   Never change these. If the best next idea needs one changed, STOP (section 5).
R7 Every change sits behind a flag whose default reproduces the old behaviour; the smoke test is
   extended to cover it; outputs go to a new OUTDIR ppo/runs/loop/iterNN_<name>/; never reuse a run dir.
R8 Research only when the idea queue is empty or the last run invalidated the queue. Use WebSearch in
   standard mode, cite papers, and tie every idea to a concrete finding in this repo. No generic theory.
R9 Report numbers faithfully. Failed runs are logged as failures with their mechanism; never quietly
   pick a better seed or window.
R10 No benchmark tuning on arm results. The synthetic generator is frozen (parameters + code hash in
   LEDGER.md) after its cheap check passes and before any arm runs. A later retune counts as a NEW
   benchmark: log it, restart that benchmark's trial count, and never compare arms across benchmarks.

==================================================================================================
3. THE PROTOCOL (one step per wake-up)
==================================================================================================
Step 0 Orient: read LEDGER.md; check run state (pgrep, tail of the iteration log). If a run is in progress,
       schedule a wake-up in 1800-3600 s and stop. If it died, relaunch it, then schedule.
Step 1 Analyse the finished run: compare_arms (+ timing score) against the relevant baselines and
       previous best; diagnostics from generations.csv (trader_exposure_mean, trader_no_trade_frac_mean,
       internal_env_fitness_max, identical_policy_pairs); check against section 4. Write a short
       Result + Mechanism entry to the ledger.
Step 2 Decide: (a) criteria met on dev -> run the sealed holdout once -> report; (b) otherwise take the
       next queued idea; (c) queue empty -> research (R8) and add 1-3 ranked ideas, each with an expected
       mechanism and the cheap check that could kill it.
Step 3 Run the cheap check (R4). If it kills the idea, log it and go back to Step 2.
Step 4 Implement the smallest change behind a flag, extend and pass the smoke test, do a 2-generation
       plumbing run, then launch with nohup into the new OUTDIR, logging to ppo/runs/loop/iterNN.log.
Step 5 Commit (message ends with the session attribution lines your system reminder gives you), update
       the ledger, schedule the next wake-up (1800-3600 s while compute runs).

Initial queue, in this order for the deadline: Task B first (it needs the compute), Task A while B runs,
Task C only if time remains after the final summary (it is NOT used in Task B's runs for this deadline).
Task A Write up the null result. Add ppo/src/signal_probe.py reproducing the section-1 checks (AUC
       table, timers, volatility targeting, timing scores per arm from runs/walkforward/per_segment.csv);
       acceptance = numbers match section 1 within +/-0.01. Add a timing score to compare_arms.py output.
       Write docs/experiment-notes.md §9e ("no exploitable timing signal -> every method converges to a
       constant exposure; coevolution's adversary sets it low").
Task C Shared-exam selection (flag --shared-exam K): after each generation's training, evaluate every
       trader on the same K uniformly drawn segments, rank PBT on that score, and credit each adversary
       with the mean over all traders on the segments it proposed. Smoke-test it. This is the rigour fix
       for selection noise. Not used in Task B's runs this time (deadline).
Task B Controlled synthetic benchmark (labelled separately; see CONSTRAINT DECISION):
       - Generator: hidden Markov regime (bull / bear / sideways) with regime-dependent drift and
         volatility at hourly bars; one or more observable features that reveal the regime with noise
         (tune the signal so a logistic probe gets AUC ~0.60-0.65, i.e. timing is possible but not
         trivial). Produce OHLCV plus the same feature pipeline, so the engine runs unchanged on it.
         ppo/tests/smoke_test.py:synthetic_df is a starting point. Store each segment's TRUE regime.
       - Shift: bear regime rare in the training pool (~10-15%), common in the dev test (~50%). Control:
         an unshifted test with the same mix as training. Use >= 200 non-overlapping test segments per
         test set, so the number of seeds, not segments, limits significance; report n@t=2 from
         compare_arms.
       - Cheap check first: the logistic timer must beat B&H on the shifted dev test; otherwise retune
         the generator. That proves the benchmark is winnable. Then FREEZE it (R10).
       - Arms, 6 seeds each, GENERATIONS=20, no --shared-exam (deadline): single, population, coevolution, coevolution_regret,
         and a new baseline `reweighted`: uniform-window training where segments are sampled with
         regime-balanced weights from the generator's TRUE regime labels (an oracle re-weighting).
         It answers the panel's question "is the adversary more than re-weighting?".
       - Hypothesis: coevolution beats single and population on the shifted test, the gap is ~0 on
         the control, its timing score is > 0, and it at least matches `reweighted` without seeing
         regime labels.
       - Fallback: if the coevolution arm collapses to cash on the benchmark (exposure < 0.2 in bull
         regimes), make coevolution_regret the primary arm before trying any other idea.

==================================================================================================
4. SUCCESS CRITERIA (pre-registered; do not edit after the first arm run)
==================================================================================================
Synthetic (primary). Let X be an arm in {coevolution, coevolution_*}. Required for any success:
  - X beats BOTH single and population on the shifted dev test: hierarchical P <= 0.05 and >= 5/6
    seed wins each;
  - on the unshifted control, the advantage is not significant (P > 0.10) or is less than a third of
    the shifted gap;
  - X's timing score is > 0;
  - confirmed ONCE on sealed seeds 9000-9099: same direction, P <= 0.10.
  Claim tiers against `reweighted`:
  - STRONG: X also beats `reweighted` (P <= 0.10). Claim: "the adversary finds a better curriculum than
    oracle regime re-weighting".
  - MODERATE: X ties `reweighted` (P > 0.10 in both directions). Claim: "coevolution recovers the oracle
    re-weighting without regime labels".
  - Otherwise (X loses to `reweighted`): not a coevolution success. Log it as "re-weighting explains the
    effect".
Real data (secondary, expected to fail): a variant beats population and single on the 48 walk-forward
segments with hierarchical P <= 0.10, timing score > 0, and DSR > 0.95 given the ledger's trial count.
Then a single check on the sealed bars from 2026-09-01.

==================================================================================================
5. STOP CONDITIONS (stop the loop, send a push notification, and write a summary into LEDGER.md)
==================================================================================================
- A synthetic STRONG or MODERATE result is met and confirmed on the sealed seeds, OR
- MAX_ITERATIONS is reached, OR it is 2026-10-09 18:00, OR
- two consecutive iterations with no improvement and no research-backed idea left, OR
- the best next idea needs a frozen real-data constraint changed (R6): stop and ask me, OR
- the same failure (crash, OOM, data error) happens twice in a row.
The final summary covers: what was tried (trial count), what worked, what failed and why, which claim
tier was reached (or the honest null-result headline if none), and the exact files and numbers to cite.

Keep chat replies to me to 3-5 lines per iteration (what ran, the result, what's next). Detail goes in the ledger.
