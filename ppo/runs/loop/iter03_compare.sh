#!/usr/bin/env bash
# compare_arms for Task C (iter03) on one synth_v1 DEV split: bash iter03_compare.sh dev_shift
# coevolution_exam vs the iter01 arms on the same split. Never run on sealed_shift (V1 sealed already used once).
# Fresh sealed sets (trial 19, once): D=synth_v1_sealed2 bash iter03_compare.sh sealed2_shift / sealed2_control
O=../runs/loop/iter01_taskB
X=../runs/loop/iter03_exam
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ../.venv/bin/python compare_arms.py \
  --data ../data/${D:-synth_v1}/$1.parquet --features ../data/synth_v1/feature_columns.pkl \
  --arms "coevolution_exam=$X/coevolution_exam_seed*/best_trader.zip" \
         "coevolution=$O/coevolution_seed*/best_trader.zip" \
         "coevolution_regret=$O/coevolution_regret_seed*/best_trader.zip" \
         "reweighted=$O/reweighted_seed*/best_trader.zip" \
         "population=$O/population_seed*/best_trader.zip" \
         "single=$O/single_seed*/best_trader.zip" \
  --final-traders --jobs ${JOBS:-6} --out $X/compare_$1.json
