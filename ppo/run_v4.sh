#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# V4: the full three-way ablation at 30 generations, 6 seeds, with the
# corrected adversary scoring and the extended parameter/behaviour logging.
#
#   nohup bash ppo/run_v4.sh > ppo/runs/v4.log 2>&1 &
#
# Everything is derived from the V3 configuration (runs/v3/*/config.json) so
# the only deliberate differences are: 30 generations instead of 20, and
# --env-skip-no-trade-matches.
#
# Override for a quick plumbing check:
#   GENERATIONS=2 SEEDS="99" TIMESTEPS=512 NTRADERS=4 NENV=4 SEGS=2 MATCHES=2 \
#     OUTDIR=../runs/v4_smoke bash ppo/run_v4.sh
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
OUTDIR="${OUTDIR:-../runs/v4}"
JOBS="${JOBS:-4}"          # concurrent runs
THREADS="${THREADS:-3}"    # torch threads each; JOBS*THREADS should be <= cores
MODES="${MODES:-coevolution population single}"

LOGDIR="$REPO/ppo/runs/$(basename "$OUTDIR")_logs"
mkdir -p "$LOGDIR"

say() { echo "[$(date +%H:%M:%S)] $*"; }

# --- 1. environment -------------------------------------------------------
if [ ! -x "$VENV/bin/python" ]; then
  say "creating $VENV (torch CPU wheels, ~200MB)"
  python3 -m venv "$VENV" || { say "FATAL: could not create venv"; exit 1; }
  "$VENV/bin/pip" install -q --upgrade pip
  # CPU index first, so the ~2GB CUDA build is never downloaded
  "$VENV/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cpu \
    || { say "FATAL: torch install failed"; exit 1; }
  "$VENV/bin/pip" install -q -r "$REPO/ppo/requirements.txt" \
    || { say "FATAL: requirements install failed"; exit 1; }
fi
if ! "$VENV/bin/python" -c "import torch, stable_baselines3, pandas, gymnasium, pyarrow, joblib" 2>/dev/null; then
  say "FATAL: $VENV is missing packages; delete it and re-run to rebuild"
  exit 1
fi
say "env ok: $("$VENV/bin/python" -c 'import torch,stable_baselines3 as s;print("torch",torch.__version__,"sb3",s.__version__)')"

# --- 2. the ablation ------------------------------------------------------
cd "$SRC" || exit 1

# V3 settings, verbatim, plus the corrected adversary scoring
CFG="--n-traders $NTRADERS --n-env $NENV --segments-per-match $SEGS \
--matches-per-trader $MATCHES --timesteps-per-match $TIMESTEPS \
--generations $GENERATIONS \
--env-weight-sigma 0.7 --env-temp-sigma 0.35 --env-fresh-frac 0.25 \
--trade-penalty 0.003 --gamma 0.997 --ent-coef-max 0.02 \
--env-skip-no-trade-matches"

run_one() {
  local mode="$1" seed="$2"
  local log="$LOGDIR/${mode}_seed${seed}.log"
  OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
    "$VENV/bin/python" run_coevolution.py --mode "$mode" --seeds "$seed" \
      $CFG --out-root "$OUTDIR" > "$log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then echo "[$(date +%H:%M:%S)]   done  ${mode} seed ${seed}"
  else echo "[$(date +%H:%M:%S)]   FAIL  ${mode} seed ${seed} (rc=$rc) -> $log"; fi
  return $rc
}
export -f run_one
export VENV CFG OUTDIR LOGDIR THREADS

NJOBS=$(for m in $MODES; do for s in $SEEDS; do echo "$m $s"; done; done | wc -l)
say "running $NJOBS jobs ($JOBS at a time, $THREADS threads each) -> $OUTDIR"
say "per-run logs in $LOGDIR"

# Slowest arm first so the long jobs start early and the queue drains evenly.
for m in $MODES; do for s in $SEEDS; do echo "$m $s"; done; done \
  | xargs -P "$JOBS" -n 2 bash -c 'run_one "$0" "$1"'

say "all runs finished"

# --- 3. evaluation on the held-out test set -------------------------------
for m in $MODES; do
  models=( $OUTDIR/${m}_seed*/best_trader.zip )
  if [ ! -e "${models[0]}" ]; then say "no models for $m, skipping eval"; continue; fi
  say "evaluating $m (${#models[@]} seeds)"
  "$VENV/bin/python" evaluate.py --models "${models[@]}" \
    --data ../data/processed/test.parquet \
    --out "$OUTDIR/${m}_eval.json" 2>&1 | sed 's/^/    /'
done

say "V4 COMPLETE"
say "  runs      : $REPO/ppo/$(echo "$OUTDIR" | sed 's|^\.\./||')"
say "  eval JSON : ${OUTDIR}/{coevolution,population,single}_eval.json"
say "  per-run   : $LOGDIR"
