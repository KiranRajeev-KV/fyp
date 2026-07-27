"""One-shot overnight orchestrator: arms-race tuning -> ablation -> evaluation.

Runs the whole strengthened-results pipeline unattended:

  Phase 1  Try a few adversary-variation settings (short, 1 seed) and AUTO-PICK
           the one that best keeps the arms race engaged (highest late-stage
           environment-population fitness -> least disengagement / collapse).
  Phase 3  Run the full three-way ablation (single/population/coevolution) with
           the chosen settings, multiple seeds, into runs/v2/ (does NOT overwrite
           the original runs/<mode>_seed* results).
  Phase 4  Evaluate each arm on the held-out test split, writing *_eval.json.

Usage (from ppo/src; nohup to survive logout):
    nohup ../.venv/bin/python run_overnight.py > ../runs/overnight.log 2>&1 &
    tail -f ../runs/overnight.log
"""

import argparse
import glob
import os
import subprocess
import sys

import numpy as np
import pandas as pd

PY = sys.executable  # the venv python running this script

# Phase 1 candidate arms-race settings (only the knobs that fight disengagement)
CANDIDATES = {
    "tuneA": {"env_weight_sigma": 0.5, "env_temp_sigma": 0.25, "timesteps": 8192, "n_env": 16},
    "tuneB": {"env_weight_sigma": 0.7, "env_temp_sigma": 0.35, "timesteps": 8192, "n_env": 16},
    "tuneC": {"env_weight_sigma": 0.5, "env_temp_sigma": 0.25, "timesteps": 4096, "n_env": 16},
}
FALLBACK = {"env_weight_sigma": 0.5, "env_temp_sigma": 0.25, "timesteps": 8192, "n_env": 16}


def run(cmd):
    print("\n>>", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd]).returncode == 0


def late_env_fitness(csv_path):
    """Mean environment-population fitness over the LAST HALF of generations.

    Higher = the adversary stayed effective (engaged); a collapse toward ~0
    is the disengagement failure mode we want to avoid.
    """
    c = pd.read_csv(csv_path)
    col = c["internal_env_fitness_max"].values
    return float(np.mean(col[len(col) // 2:]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46, 47])
    ap.add_argument("--generations", type=int, default=20)
    ap.add_argument("--tune-generations", type=int, default=12,
                    help="shorter runs for the Phase 1 tuning sweep")
    ap.add_argument("--segments-per-match", type=int, default=6)
    ap.add_argument("--n-traders", type=int, default=8)
    ap.add_argument("--out-root", default="../runs/v2")
    ap.add_argument("--test-data", default="../data/processed/test.parquet")
    ap.add_argument("--skip-tuning", action="store_true",
                    help="use FALLBACK settings and skip Phase 1")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)

    # ---------------- Phase 1: arms-race tuning ----------------
    chosen = FALLBACK
    if not args.skip_tuning:
        print("=" * 60)
        print("PHASE 1 - arms-race tuning (coevolution, 1 seed each)")
        print("=" * 60)
        scores = {}
        for name, cfg in CANDIDATES.items():
            out = f"../runs/tune/{name}"
            ok = run([PY, "run_coevolution.py", "--mode", "coevolution", "--seeds", 42,
                      "--generations", args.tune_generations,
                      "--segments-per-match", args.segments_per_match,
                      "--n-env", cfg["n_env"],
                      "--env-weight-sigma", cfg["env_weight_sigma"],
                      "--env-temp-sigma", cfg["env_temp_sigma"],
                      "--timesteps-per-match", cfg["timesteps"],
                      "--out-root", out])
            csv = f"{out}/coevolution_seed42/generations.csv"
            if ok and os.path.exists(csv):
                scores[name] = late_env_fitness(csv)
                print(f"  [{name}] late-stage env fitness = {scores[name]:+.3f}")
            else:
                print(f"  [{name}] FAILED - skipping")
        if scores:
            best = max(scores, key=scores.get)
            chosen = CANDIDATES[best]
            print(f"\nPhase 1 winner: {best}  {chosen}")
        else:
            print("\nAll tuning runs failed; falling back to defaults:", FALLBACK)

    print("\nChosen arms-race settings:", chosen)

    # ---------------- Phase 3: full ablation ----------------
    print("=" * 60)
    print(f"PHASE 3 - three-way ablation, seeds {args.seeds} -> {args.out_root}")
    print("=" * 60)
    for mode in ["single", "population", "coevolution"]:
        run([PY, "run_coevolution.py", "--mode", mode, "--seeds", *args.seeds,
             "--generations", args.generations,
             "--n-traders", args.n_traders,
             "--segments-per-match", args.segments_per_match,
             "--n-env", chosen["n_env"],
             "--env-weight-sigma", chosen["env_weight_sigma"],
             "--env-temp-sigma", chosen["env_temp_sigma"],
             "--timesteps-per-match", chosen["timesteps"],
             "--out-root", args.out_root])

    # ---------------- Phase 4: evaluation ----------------
    print("=" * 60)
    print("PHASE 4 - evaluate each arm on the held-out test split")
    print("=" * 60)
    for mode in ["single", "population", "coevolution"]:
        models = sorted(glob.glob(f"{args.out_root}/{mode}_seed*/best_trader.zip"))
        if not models:
            print(f"  [{mode}] no models found - skipping")
            continue
        run([PY, "evaluate.py", "--models", *models,
             "--data", args.test_data,
             "--out", f"{args.out_root}/{mode}_eval.json"])

    print("\nDONE. Eval JSONs in", args.out_root)
    print("Send me the *_eval.json files and I'll regenerate the figure + notes.")


if __name__ == "__main__":
    main()
