#!/usr/bin/env bash
# compare_arms for iter01 on one synth_v1 split: bash iter01_compare.sh dev_shift
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../src" && pwd)"
cd "$SRC" || exit 1
O=../runs/loop/iter01_taskB
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ../.venv/bin/python compare_arms.py \
  --data ../data/synth_v1/$1.parquet --features ../data/synth_v1/feature_columns.pkl \
  --arms "coevolution=$O/coevolution_seed*/best_trader.zip" \
         "coevolution_regret=$O/coevolution_regret_seed*/best_trader.zip" \
         "reweighted=$O/reweighted_seed*/best_trader.zip" \
         "population=$O/population_seed*/best_trader.zip" \
         "single=$O/single_seed*/best_trader.zip" \
  --final-traders --jobs ${JOBS:-6} --out $O/compare_$1.json
