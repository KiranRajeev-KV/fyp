#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# V5: the ablation re-run on the audited engine.
#
#   nohup bash ppo/run_v5.sh >> ppo/runs/v5.log 2>&1 &
#
# Safe to relaunch after a shutdown: finished runs are skipped and partial
# runs resume from their per-generation checkpoint (see ppo/run_all.sh).
#
# Same configuration as V4 (30 generations, 6 seeds, corrected adversary
# scoring) plus the audit fixes:
#   * trader hall of fame is a real snapshot (V4's env-HOF loop overwrote
#     trader 0 every generation in the coevolution arm only)
#   * every final trader is saved (final_traders/), not just rank 0
#   * baselines pay the same slippage as the agent
#   * config.json records the git revision
#   * out_dir overwrite guard (a V4 relaunch destroyed 16/18 ledgers)
# and a FOURTH arm, coevolution_regret: identical to coevolution except the
# adversary is scored on regret vs the best trivial policy per segment
# instead of -Sharpe (removes the fitness ceiling at 0 that ends the arms race).
#
# Override for a quick plumbing check:
#   GENERATIONS=2 SEEDS="99" TIMESTEPS=512 NTRADERS=4 NENV=4 SEGS=2 MATCHES=2 \
#     OUTDIR=../runs/v5_smoke bash ppo/run_v5.sh
# ---------------------------------------------------------------------------
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO/ppo/src"
VENV="$REPO/ppo/.venv"

GENERATIONS="${GENERATIONS:-30}"
SEEDS="${SEEDS:-42 43 44 45 46 47}"
TIMESTEPS="${TIMESTEPS:-8192}"
NTRADERS="${NTRADERS:-8}"
NENV="${NENV:-16}"
SEGS="${SEGS:-6}"
MATCHES="${MATCHES:-3}"
OUTDIR="${OUTDIR:-../runs/v5}"
JOBS="${JOBS:-4}"          # concurrent runs
THREADS="${THREADS:-3}"    # torch threads each; JOBS*THREADS should be <= cores
# arms: engine mode, optionally suffixed with a variant name
ARMS="${ARMS:-coevolution_regret coevolution population single}"
FORCE="${FORCE:-0}"
# data overrides (Task B synthetic benchmark); defaults = the real BTC splits
TRAIN_DATA="${TRAIN_DATA:-../data/processed/train.parquet}"
VAL_DATA="${VAL_DATA:-../data/processed/val.parquet}"
FEATURES="${FEATURES:-../models/feature_columns.pkl}"
EVAL="${EVAL:-1}"          # 0 = skip the real-data test evaluation at the end

LOGDIR="$REPO/ppo/runs/$(basename "$OUTDIR")_logs"
mkdir -p "$LOGDIR"

say() { echo "[$(date +%H:%M:%S)] $*"; }

# --- 0. idempotent: finished runs are skipped, partial runs resume ----------
# Every generation is checkpointed (out_dir/checkpoint/), so after a shutdown
# just relaunch this script: complete runs (best_trader.zip present) are
# skipped, partial ones continue from their last generation, and the engine
# refuses to touch a populated run that has no checkpoint unless FORCE=1.
cd "$SRC" || exit 1

# --- 1. environment -------------------------------------------------------
if ! "$VENV/bin/python" -c "import torch, stable_baselines3, pandas, gymnasium, pyarrow, joblib, sklearn" 2>/dev/null; then
  say "FATAL: $VENV is missing packages (see run_v4.sh for the install recipe)"
  exit 1
fi
say "env ok: $("$VENV/bin/python" -c 'import torch,stable_baselines3 as s;print("torch",torch.__version__,"sb3",s.__version__)')"
say "git: $(git -C "$REPO" rev-parse --short HEAD)$(git -C "$REPO" diff --quiet HEAD || echo '-dirty')"

# --- 2. the ablation ------------------------------------------------------
# V4 settings verbatim
CFG="--n-traders $NTRADERS --n-env $NENV --segments-per-match $SEGS \
--matches-per-trader $MATCHES --timesteps-per-match $TIMESTEPS \
--generations $GENERATIONS \
--env-weight-sigma 0.7 --env-temp-sigma 0.35 --env-fresh-frac 0.25 \
--trade-penalty 0.003 --gamma 0.997 --ent-coef-max 0.02"
CFG="$CFG --resume --train-data $TRAIN_DATA --val-data $VAL_DATA --features $FEATURES"
[ "$FORCE" = "1" ] && CFG="$CFG --overwrite"

arm_flags() {
  case "$1" in
    coevolution)        echo "--mode coevolution --env-skip-no-trade-matches" ;;
    coevolution_regret) echo "--mode coevolution --env-fitness regret --run-name coevolution_regret" ;;
    # the audit's two trader-side fixes together: half the segments drawn
    # uniformly so bull coverage survives, no trade penalty and an entropy floor
    # so the policy can re-enter the market after the bear phase
    coevolution_mixed)  echo "--mode coevolution --env-skip-no-trade-matches \
--env-adversary-frac 0.5 --trade-penalty 0 --ent-coef-min 0.005 --run-name coevolution_mixed" ;;
    population)         echo "--mode population" ;;
    # oracle baseline for the synthetic benchmark: population + PBT, segments
    # drawn regime-balanced on the TRUE regime labels (no adversary)
    reweighted)         echo "--mode population --segment-sampling regime_balanced --run-name reweighted" ;;
    # Task C: coevolution with shared-exam selection (all traders ranked on the
    # same 16 uniform segments; adversaries credited over all traders)
    coevolution_exam)   echo "--mode coevolution --env-skip-no-trade-matches --shared-exam 16 --run-name coevolution_exam" ;;
    single)             echo "--mode single" ;;
    *) echo "unknown arm $1" >&2; return 1 ;;
  esac
}

run_one() {
  local arm="$1" seed="$2"
  local log="$LOGDIR/${arm}_seed${seed}.log"
  if [ -e "$OUTDIR/${arm}_seed${seed}/best_trader.zip" ] && [ "$FORCE" != "1" ]; then
    echo "[$(date +%H:%M:%S)]   skip  ${arm} seed ${seed} (done)"; return 0
  fi
  local flags; flags="$(arm_flags "$arm")" || return 1
  OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
    "$VENV/bin/python" run_coevolution.py --seeds "$seed" \
      $CFG $flags --out-root "$OUTDIR" >> "$log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then echo "[$(date +%H:%M:%S)]   done  ${arm} seed ${seed}"
  else echo "[$(date +%H:%M:%S)]   FAIL  ${arm} seed ${seed} (rc=$rc) -> $log"; fi
  return $rc
}
export -f run_one arm_flags
export VENV CFG OUTDIR LOGDIR THREADS

NJOBS=$(for a in $ARMS; do for s in $SEEDS; do echo "$a $s"; done; done | wc -l)
say "running $NJOBS jobs ($JOBS at a time, $THREADS threads each) -> $OUTDIR"
say "per-run logs in $LOGDIR"

# Slowest arms first so the long jobs start early and the queue drains evenly.
for a in $ARMS; do for s in $SEEDS; do echo "$a $s"; done; done \
  | xargs -P "$JOBS" -n 2 bash -c 'run_one "$0" "$1"'

say "all runs finished"
[ "$EVAL" = "1" ] || { say "EVAL=0: skipping test evaluation"; exit 0; }

# --- 3. evaluation on the held-out test set -------------------------------
ARMSPEC=""
for a in $ARMS; do
  if ls "$OUTDIR"/${a}_seed*/best_trader.zip >/dev/null 2>&1; then
    ARMSPEC="$ARMSPEC ${a}=$OUTDIR/${a}_seed*/best_trader.zip"
  else
    say "no models for $a, skipping"
  fi
done
for split in test test_ext; do
  data="../data/processed/${split}.parquet"
  [ -e "$data" ] || { say "no $data, skipping"; continue; }
  say "compare_arms on $split (rank-0 and final-population mean)"
  "$VENV/bin/python" compare_arms.py --data "$data" --arms $ARMSPEC --final-traders \
      --out "$OUTDIR/compare_${split}.json" 2>&1 | grep -v '^  ' | sed 's/^/    /'
done
# legacy per-arm evaluate.py JSONs (same format as V1-V4) for the old figure scripts
for a in $ARMS; do
  models=( $OUTDIR/${a}_seed*/best_trader.zip )
  [ -e "${models[0]}" ] || continue
  "$VENV/bin/python" evaluate.py --models "${models[@]}" \
    --data ../data/processed/test.parquet --out "$OUTDIR/${a}_eval.json" > /dev/null 2>&1
done

say "V5 COMPLETE"
say "  runs      : $REPO/ppo/$(echo "$OUTDIR" | sed 's|^\.\./||')"
say "  compare   : ${OUTDIR}/compare_test.json (+ compare_test_ext.json if test_ext exists)"
say "  per-run   : $LOGDIR"
