#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Walk-forward ablation, sharded: one process per (fold, arm, seed), JOBS at a
# time, then a single aggregation pass.
#
#   nohup bash ppo/run_walkforward.sh >> ppo/runs/walkforward.log 2>&1 &
#
# Safe to relaunch after a shutdown: finished shards are skipped and partial
# ones resume from their per-generation checkpoint (see ppo/run_all.sh).
#
# Uses the SAME engine flags as run_v5.sh (via engine_cli.py) so every fold's
# config.json matches V5's except mode/seed. Default 20 generations (the V3/V4
# finding: 30 bought nothing over 20) and 3 seeds for the population arms;
# the single arm is cheap, so it gets all 6.
#
# Fold geometry (1h bars): train 40,000 / val tail 4,000 / test 6,000, stepping
# 6,000 -> 6 folds whose test windows run Mar 2022 .. Apr 2026 (bear, bull and
# sideways), 8 non-overlapping 720-bar test segments each = 48 in total.
#
# Plumbing check:
#   GENERATIONS=1 SEEDS="99" SINGLE_SEEDS="99" TIMESTEPS=512 NTRADERS=3 NENV=4 \
#     SEGS=2 MATCHES=2 FOLDS="0" OUTDIR=../runs/wf_smoke bash ppo/run_walkforward.sh
# ---------------------------------------------------------------------------
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO/ppo/src"
VENV="$REPO/ppo/.venv"

GENERATIONS="${GENERATIONS:-20}"
SEEDS="${SEEDS:-42 43 44}"
SINGLE_SEEDS="${SINGLE_SEEDS:-42 43 44 45 46 47}"
TIMESTEPS="${TIMESTEPS:-8192}"
NTRADERS="${NTRADERS:-8}"
NENV="${NENV:-16}"
SEGS="${SEGS:-6}"
MATCHES="${MATCHES:-3}"
OUTDIR="${OUTDIR:-../runs/walkforward}"
JOBS="${JOBS:-4}"
THREADS="${THREADS:-3}"
ARMS="${ARMS:-coevolution population single}"    # add coevolution_regret for the 4-arm version
FOLDS="${FOLDS:-0 1 2 3 4 5}"
TRAIN_SPAN="${TRAIN_SPAN:-40000}"
TEST_SPAN="${TEST_SPAN:-6000}"
VAL_SPAN="${VAL_SPAN:-4000}"

LOGDIR="$REPO/ppo/runs/$(basename "$OUTDIR")_logs"
mkdir -p "$LOGDIR"
say() { echo "[$(date +%H:%M:%S)] $*"; }

cd "$SRC" || exit 1
if ! "$VENV/bin/python" -c "import torch, stable_baselines3, pandas, gymnasium, pyarrow, joblib, sklearn" 2>/dev/null; then
  say "FATAL: $VENV is missing packages"; exit 1
fi
say "git: $(git -C "$REPO" rev-parse --short HEAD)$(git -C "$REPO" diff --quiet HEAD || echo '-dirty')"

GEOM="--train-span $TRAIN_SPAN --test-span $TEST_SPAN --val-span $VAL_SPAN --out-root $OUTDIR"
# V5 engine settings, verbatim
CFG="--n-traders $NTRADERS --n-env $NENV --segments-per-match $SEGS \
--matches-per-trader $MATCHES --timesteps-per-match $TIMESTEPS \
--generations $GENERATIONS \
--env-weight-sigma 0.7 --env-temp-sigma 0.35 --env-fresh-frac 0.25 \
--trade-penalty 0.003 --gamma 0.997 --ent-coef-max 0.02 --resume"

arm_flags() {
  case "$1" in
    coevolution)        echo "--modes coevolution --env-skip-no-trade-matches" ;;
    coevolution_regret) echo "--modes coevolution:coevolution_regret --env-fitness regret" ;;
    population)         echo "--modes population" ;;
    single)             echo "--modes single" ;;
    *) echo "unknown arm $1" >&2; return 1 ;;
  esac
}

run_one() {
  local fold="$1" arm="$2" seed="$3"
  local log="$LOGDIR/fold${fold}_${arm}_seed${seed}.log"
  # skip finished shards so the driver can be re-launched after an interruption
  if [ -e "$OUTDIR/fold${fold}/${arm}_seed${seed}/best_trader.zip" ]; then
    echo "[$(date +%H:%M:%S)]   skip  fold $fold ${arm} seed ${seed} (done)"; return 0
  fi
  local flags; flags="$(arm_flags "$arm")" || return 1
  OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
    "$VENV/bin/python" run_walkforward.py --folds "$fold" $flags --seeds "$seed" \
      $GEOM $CFG >> "$log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then echo "[$(date +%H:%M:%S)]   done  fold $fold ${arm} seed ${seed}"
  else echo "[$(date +%H:%M:%S)]   FAIL  fold $fold ${arm} seed ${seed} (rc=$rc) -> $log"; fi
  return $rc
}
export -f run_one arm_flags
export VENV CFG GEOM OUTDIR LOGDIR THREADS

# fold 0's test_scaled.parquet etc. are written by whichever shard runs first;
# writes are idempotent (only-if-absent) so concurrent shards are safe.
jobs_list() {
  for f in $FOLDS; do for a in $ARMS; do
    if [ "$a" = "single" ]; then seeds="$SINGLE_SEEDS"; else seeds="$SEEDS"; fi
    for s in $seeds; do echo "$f $a $s"; done
  done; done
}
NJOBS=$(jobs_list | wc -l)
say "running $NJOBS shards ($JOBS at a time, $THREADS threads each) -> $OUTDIR"
jobs_list | xargs -P "$JOBS" -n 3 bash -c 'run_one "$0" "$1" "$2"'
say "all shards finished"

say "aggregating"
"$VENV/bin/python" run_walkforward.py --aggregate-only $GEOM $CFG 2>&1 | grep -v Warn | sed 's/^/    /'

ARMSPEC=""
for a in $ARMS; do ARMSPEC="$ARMSPEC ${a}=${a}_seed*/best_trader.zip"; done
say "compare_arms across folds"
"$VENV/bin/python" compare_arms.py --walkforward-root "$OUTDIR" --arms $ARMSPEC --final-traders \
    --out "$OUTDIR/compare.json" 2>&1 | grep -v '^  ' | sed 's/^/    /'
say "WALK-FORWARD COMPLETE -> $OUTDIR/compare.json"
