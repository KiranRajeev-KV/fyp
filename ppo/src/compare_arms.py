"""Arm-vs-arm comparison for the ablation -- the test the paper's claim needs.

evaluate.py tests one arm against buy-and-hold and MA-crossover. The
hypothesis is coevolution vs single-agent PPO and vs population-only, so this
script evaluates every arm on the same held-out segments and reports, for
every ordered pair (arm, reference), P(arm <= reference) under three paired
bootstraps that answer different questions:

  segments      resample the held-out segments; each segment's value is the
                mean over seeds.  "Does the edge generalise to new windows?"
                (n = number of segments; underpowered at n = 9)
  seeds         resample seeds; each seed's value is the mean over segments.
                "Is the ordering reproducible across training runs on this
                window?"  (n = number of seeds; pairs arms on equal seeds)
  hierarchical  resample seeds and segments jointly -- the headline number.

Per-seed model: by default the saved best_trader.zip (final-generation
internal rank 0, which is one noisy draw from the final population). With
--final-traders every trader in final_traders/ is evaluated too and each seed
is represented by its population MEAN (rank-0 is still reported).

Examples below run from ppo/src; quote name=glob arguments for zsh.

Single split:
    python compare_arms.py --data ../data/processed/test.parquet \
        --arms "coevolution=../runs/v5/coevolution_seed*/best_trader.zip" \
               "population=../runs/v5/population_seed*/best_trader.zip" \
               "single=../runs/v5/single_seed*/best_trader.zip" \
        --final-traders --out ../runs/v5/compare.json

Walk-forward (globs are relative to each fold directory; segments pool across folds):
    python compare_arms.py --walkforward-root ../runs/walkforward \
        --arms "coevolution=coevolution_seed*/best_trader.zip" \
               "single=single_seed*/best_trader.zip" --out ../runs/walkforward/compare.json
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

from segments import build_segment_pool
import evaluate as ev
from project_paths import project_path

BASELINES = ["buy_hold", "ma_crossover"]


def seed_of(path):
    m = re.search(r"_seed(\d+)", path)
    return int(m.group(1)) if m else -1


def evaluate_split(df, arms, env_kwargs, feature_columns, segment_length,
                   segment_stride, final_traders, fold=0, seg_offset=0, jobs=1):
    """Long-format rows: one per (arm, seed, model_kind, trader, segment)."""
    pool = build_segment_pool(df, segment_length=segment_length, stride=segment_stride)
    rows = []
    base = ev.eval_baselines_per_segment(
        pool, env_kwargs["initial_balance"], env_kwargs["trading_fee"],
        env_kwargs.get("baseline_slippage", env_kwargs.get("slippage", 0.0)))
    env_kwargs = {k: v for k, v in env_kwargs.items() if k != "baseline_slippage"}
    for i, (b, s) in enumerate(zip(base, pool.segments)):
        for name in BASELINES:
            rows.append({"arm": name, "seed": -1, "kind": "baseline", "trader": -1,
                         "fold": fold, "segment": seg_offset + i, "regime": s.regime,
                         "move_regime": s.move_regime,
                         "start": s.start, **{k: b[name][k] for k in ev.METRIC_KEYS},
                         "trades": np.nan, "exposure": np.nan})

    jobs_list = []
    for arm, paths in arms.items():
        for p in paths:
            seed = seed_of(p)
            jobs_list.append((arm, seed, "best", -1, p))
            if final_traders:
                fdir = os.path.join(os.path.dirname(p), "final_traders")
                for q in sorted(glob.glob(os.path.join(fdir, "trader_*.zip"))):
                    tid = int(re.search(r"trader_(\d+)\.zip", q).group(1))
                    jobs_list.append((arm, seed, "final", tid, q))
    for arm, seed, kind, tid, q in jobs_list:
        print(f"  {arm:16s} seed {seed:3d} {kind:5s} {os.path.relpath(q)}", flush=True)
    from joblib import Parallel, delayed
    results = Parallel(n_jobs=jobs)(
        delayed(ev.eval_model_per_segment)(q, pool, env_kwargs, feature_columns)
        for *_, q in jobs_list)
    for (arm, seed, kind, tid, q), per_seg in zip(jobs_list, results):
        for i, (r, s) in enumerate(zip(per_seg, pool.segments)):
            rows.append({"arm": arm, "seed": seed, "kind": kind, "trader": tid,
                         "fold": fold, "segment": seg_offset + i,
                         "regime": s.regime, "move_regime": s.move_regime,
                         "start": s.start,
                         **{k: r[k] for k in ev.METRIC_KEYS},
                         "trades": r["trades"], "exposure": r["exposure"]})
    return rows, len(pool)


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def seed_matrix(long, arm, kind, metric="sharpe"):
    """(seeds x segments) matrix of per-seed values; for kind='final' the
    per-seed value is the mean over that seed's final population."""
    d = long[(long.arm == arm) & (long.kind == kind)]
    m = d.groupby(["seed", "segment"])[metric].mean().unstack("segment").sort_index()
    return m


def baseline_matrix(long, name, seeds, metric="sharpe"):
    d = long[long.arm == name].groupby("segment")[metric].mean().sort_index()
    return pd.DataFrame([d.values] * len(seeds), index=seeds, columns=d.index)


def bootstrap_pair(A, B, rng, n_boot=20000):
    """A, B: aligned (seeds x segments) DataFrames. Returns the three P(A <= B)."""
    A, B = A.align(B, join="inner")
    D = (A - B).values
    S, G = D.shape
    out = {"n_seeds": int(S), "n_segments": int(G),
           "mean_diff": float(D.mean()),
           "seed_wins": int((D.mean(axis=1) > 0).sum())}
    seg = D.mean(axis=0)
    out["p_segments"] = float((rng.choice(seg, size=(n_boot, G)).mean(axis=1) <= 0).mean())
    sd = D.mean(axis=1)
    out["p_seeds"] = float((rng.choice(sd, size=(n_boot, S)).mean(axis=1) <= 0).mean()) if S > 1 else float("nan")
    si = rng.integers(0, S, size=(n_boot, S))
    gi = rng.integers(0, G, size=(n_boot, G))
    hier = np.array([D[si[b]][:, gi[b]].mean() for b in range(n_boot)])
    out["p_hierarchical"] = float((hier <= 0).mean())
    # sample size needed for |t| = 2 on the per-segment differences
    if G > 1 and seg.mean() != 0:
        out["n_segments_for_t2"] = int(np.ceil((2 * seg.std(ddof=1) / seg.mean()) ** 2))
    return out


def timing_score(long, arm, kind):
    """Does the arm time the market, or just hold a constant exposure?

    timing: mean exposure on segments where buy-and-hold made money minus mean
    exposure where it lost (per seed, then averaged); corr: corr(exposure,
    buy-and-hold return) pooled over all (seed, segment) cells. Both are ~0 for
    a constant-exposure policy whatever its level.
    """
    E = seed_matrix(long, arm, kind, "exposure")
    move = long[long.arm == "buy_hold"].groupby("segment")["total_return_pct"].mean()
    move = move.reindex(E.columns).values
    up, down = move > 0, move < 0
    diff = [row[up].mean() - row[down].mean() for row in E.values]
    flat, moves = E.values.ravel(), np.tile(move, len(E))
    corr = np.corrcoef(flat, moves)[0, 1] if flat.std() > 0 else 0.0
    return {"timing": float(np.mean(diff)), "corr_exposure_move": float(corr),
            "exposure": float(E.values.mean()),
            "exposure_up": float(E.values[:, up].mean()),
            "exposure_down": float(E.values[:, down].mean())}


def summarize_arm(long, arm, kind, metric="sharpe"):
    M = seed_matrix(long, arm, kind, metric)
    per_seed = M.mean(axis=1)
    out = {"mean": float(per_seed.mean()), "std_seeds": float(per_seed.std(ddof=0)),
           "n_seeds": int(len(per_seed)), "per_seed": {int(s): float(v) for s, v in per_seed.items()}}
    d = long[(long.arm == arm) & (long.kind == kind)]
    out["per_regime"] = {r: float(g[metric].mean()) for r, g in d.groupby("regime")}
    out["per_move_regime"] = {r: float(g[metric].mean()) for r, g in d.groupby("move_regime")}
    out["per_regime_exposure"] = {r: float(g["exposure"].mean()) for r, g in d.groupby("regime")}
    out.update(timing_score(long, arm, kind))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", required=True,
                    help="name=glob pairs; glob matches best_trader.zip files")
    ap.add_argument("--data", default=None, help="held-out parquet (single split)")
    ap.add_argument("--walkforward-root", default=None,
                    help="root with fold*/test_scaled.parquet; arm globs are relative to each fold dir")
    ap.add_argument("--features", default=project_path("models/feature_columns.pkl"))
    ap.add_argument("--segment-length", type=int, default=720)
    ap.add_argument("--segment-stride", type=int, default=720)
    ap.add_argument("--initial-balance", type=float, default=10000.0)
    ap.add_argument("--trading-fee", type=float, default=0.001)
    ap.add_argument("--slippage", type=float, default=0.0005,
                    help="per-side slippage the AGENT pays in TradingEnv")
    ap.add_argument("--baseline-slippage", type=float, default=None,
                    help="slippage charged to buy-and-hold / MA-crossover "
                         "(default: same as --slippage; 0 reproduces the V1-V4 baseline numbers)")
    ap.add_argument("--final-traders", action="store_true",
                    help="also evaluate final_traders/*.zip and use per-seed population mean")
    ap.add_argument("--metric", default="sharpe", choices=ev.METRIC_KEYS)
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--jobs", type=int, default=1, help="models evaluated in parallel")
    ap.add_argument("--out", default=None, help="JSON output")
    ap.add_argument("--csv", default=None, help="long-format per-segment CSV (default: <out>.csv)")
    args = ap.parse_args()
    if bool(args.data) == bool(args.walkforward_root):
        ap.error("give exactly one of --data or --walkforward-root")

    import joblib
    feature_columns = joblib.load(args.features)
    env_kwargs = {"initial_balance": args.initial_balance,
                  "trading_fee": args.trading_fee, "slippage": args.slippage,
                  "baseline_slippage": (args.slippage if args.baseline_slippage is None
                                        else args.baseline_slippage)}
    arm_globs = dict(a.split("=", 1) for a in args.arms)

    rows = []
    if args.data:
        arms = {n: sorted(glob.glob(g)) for n, g in arm_globs.items()}
        for n, ps in arms.items():
            if not ps:
                ap.error(f"arm {n}: no models match {arm_globs[n]}")
        df = pd.read_parquet(args.data)
        rows, n = evaluate_split(df, arms, env_kwargs, feature_columns,
                                 args.segment_length, args.segment_stride, args.final_traders,
                                 jobs=args.jobs)
    else:
        offset = 0
        fold_dirs = [d for d in glob.glob(os.path.join(args.walkforward_root, "fold[0-9]*"))
                     if os.path.isdir(d) and re.fullmatch(r"fold\d+", os.path.basename(d))]
        for fdir in sorted(fold_dirs, key=lambda p: int(re.search(r"fold(\d+)", p).group(1))):
            tp = os.path.join(fdir, "test_scaled.parquet")
            if not os.path.exists(tp):
                continue
            k = int(re.search(r"fold(\d+)", fdir).group(1))
            arms = {n: sorted(glob.glob(os.path.join(fdir, g))) for n, g in arm_globs.items()}
            arms = {n: ps for n, ps in arms.items() if ps}
            if not arms:
                continue
            print(f"fold {k}: {', '.join(f'{n}({len(ps)})' for n, ps in arms.items())}")
            r, n = evaluate_split(pd.read_parquet(tp), arms, env_kwargs, feature_columns,
                                  args.segment_length, args.segment_stride,
                                  args.final_traders, fold=k, seg_offset=offset,
                                  jobs=args.jobs)
            rows += r
            offset += n

    long = pd.DataFrame(rows)
    csv_path = args.csv or (args.out.rsplit(".", 1)[0] + ".csv" if args.out else None)
    if csv_path:
        long.to_csv(csv_path, index=False)

    rng = np.random.default_rng(0)
    kinds = ["best"] + (["final"] if args.final_traders else [])
    arm_names = list(arm_globs)
    results = {"metric": args.metric, "n_segments": int(long.segment.nunique()),
               "regimes": long[long.arm == BASELINES[0]].regime.value_counts().to_dict(),
               "move_regimes": long[long.arm == BASELINES[0]].move_regime.value_counts().to_dict(),
               "arms": {}, "baselines": {}, "pairs": {}}
    for name in BASELINES:
        d = long[long.arm == name]
        results["baselines"][name] = {
            "mean": float(d[args.metric].mean()),
            "per_regime": {r: float(g[args.metric].mean()) for r, g in d.groupby("regime")},
            "per_move_regime": {r: float(g[args.metric].mean())
                                for r, g in d.groupby("move_regime")}}
    for kind in kinds:
        results["arms"][kind] = {a: summarize_arm(long, a, kind, args.metric) for a in arm_names}
        results["pairs"][kind] = {}
        for a in arm_names:
            A = seed_matrix(long, a, kind, args.metric)
            refs = [(b, seed_matrix(long, b, kind, args.metric)) for b in arm_names if b != a]
            refs += [(b, baseline_matrix(long, b, A.index, args.metric)) for b in BASELINES]
            for b, B in refs:
                results["pairs"][kind][f"{a} vs {b}"] = bootstrap_pair(A, B, rng, args.n_boot)

    # ---- report ----
    print(f"\nmetric: {args.metric}   segments: {results['n_segments']}  "
          f"regimes (trailing-label majority): {results['regimes']}  "
          f"(by segment's own move): {results['move_regimes']}")
    for kind in kinds:
        label = "rank-0 best_trader" if kind == "best" else "final-population mean"
        print(f"\n== per-seed model: {label} ==")
        print(f"{'arm':16s}{'mean':>9s}{'sd(seeds)':>11s}{'n':>4s}   "
              f"per-regime (trailing labels) | (segment move)")
        fmt = lambda d: "  ".join(f"{r} {v:+.2f}" for r, v in sorted(d.items()))
        for a in arm_names:
            s = results["arms"][kind][a]
            print(f"{a:16s}{s['mean']:+9.3f}{s['std_seeds']:11.3f}{s['n_seeds']:4d}   "
                  f"{fmt(s['per_regime'])} | {fmt(s['per_move_regime'])}")
        for b in BASELINES:
            s = results["baselines"][b]
            print(f"{b:16s}{s['mean']:+9.3f}{'':11s}{'':4s}   "
                  f"{fmt(s['per_regime'])} | {fmt(s['per_move_regime'])}")
        print(f"\n{'timing (R5)':16s}{'timing':>9s}{'corr':>7s}{'expo':>7s}{'up':>7s}{'down':>7s}"
              f"   exposure per regime")
        for a in arm_names:
            s = results["arms"][kind][a]
            print(f"{a:16s}{s['timing']:+9.3f}{s['corr_exposure_move']:+7.3f}{s['exposure']:7.2f}"
                  f"{s['exposure_up']:7.2f}{s['exposure_down']:7.2f}   "
                  + "  ".join(f"{r} {v:.2f}" for r, v in sorted(s['per_regime_exposure'].items())))
        print(f"\n{'P(A <= B)':34s}{'segments':>10s}{'seeds':>8s}{'hier.':>8s}"
              f"{'diff':>8s}{'wins':>7s}{'n@t=2':>7s}")
        for pair, r in results["pairs"][kind].items():
            print(f"{pair:34s}{r['p_segments']:10.3f}{r['p_seeds']:8.3f}"
                  f"{r['p_hierarchical']:8.3f}{r['mean_diff']:+8.3f}"
                  f"{r['seed_wins']:4d}/{r['n_seeds']:<3d}{r.get('n_segments_for_t2', float('nan')):7.0f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwritten {args.out}" + (f" and {csv_path}" if csv_path else ""))


if __name__ == "__main__":
    main()
