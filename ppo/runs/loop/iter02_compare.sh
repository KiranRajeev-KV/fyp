#!/usr/bin/env bash
# compare_arms for iter02 on one synth_v2 split: bash iter02_compare.sh dev_shift
O=../runs/loop/iter02_v2
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ../.venv/bin/python compare_arms.py \
  --data ../data/synth_v2/$1.parquet --features ../data/synth_v2/feature_columns.pkl \
  --arms "coevolution=$O/coevolution_seed*/best_trader.zip" \
         "coevolution_regret=$O/coevolution_regret_seed*/best_trader.zip" \
         "reweighted=$O/reweighted_seed*/best_trader.zip" \
         "population=$O/population_seed*/best_trader.zip" \
         "single=$O/single_seed*/best_trader.zip" \
  --final-traders --jobs ${JOBS:-6} --out $O/compare_$1.json
