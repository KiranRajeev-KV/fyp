"""Historical Phase 2 P1: shift dose-response using SYNTH_V1 parameters (no retraining).

build: one pool of 500 fresh segments (seeds 3000-3249, 2 x 720 bars each, shifted
       mix, V1 params and V1 train scaler; never the sealed 9000-9099) -> pool.parquet.
       Evaluate it once with compare_arms (--csv) to get per-segment results.
analyse: for each target bear share pick 200 segments by their TRUE regime
       (nested: the first k bear segments, the rest bull/sideways in the train
       ratio 47:43), then bootstrap coevolution vs single / population / reweighted.

Examples from ppo/src:
    python synth_dose.py build --out ../data/synth_v1_dose
    python synth_dose.py analyse --csv <compare csv> --out <json> --fig ../figures/synth_dose_response.png
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd

import synth_market as sm
from compare_arms import bootstrap_pair, seed_matrix, timing_score
from project_paths import project_path

SHARES = [0.13, 0.25, 0.40, 0.58, 0.75]
N_SEG = 200
ARMS = ["coevolution", "coevolution_regret", "reweighted", "population", "single"]


def build(out, first_seed=3000, n_blocks=250):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "pool.parquet")
    if os.path.exists(path):
        raise FileExistsError(path)
    p = sm.PARAMS
    df = pd.concat([sm.series(p["block_bars"], p["shift_mix"], first_seed + i)
                    for i in range(n_blocks)], ignore_index=True)
    scaler = joblib.load(project_path("data/synth_v1/feature_scaler.pkl"))
    df.loc[:, sm.FEATURES] = scaler.transform(df[sm.FEATURES])
    df.to_parquet(path, index=False)
    regs = pd.Series(sm.segment_regimes(df)).value_counts().to_dict()
    print(f"{len(df)} bars, segment regimes {regs}")


def select(labels, share, n=N_SEG, train_mix=(0.47, 0.43)):
    """labels: Series segment -> true regime. Nested picks in segment order."""
    k = round(share * n)
    nb = n - k
    n_bull = round(nb * train_mix[0] / sum(train_mix))
    pick = lambda r, m: list(labels[labels == r].index[:m])
    segs = pick("bear", k) + pick("bull", n_bull) + pick("sideways", nb - n_bull)
    assert len(segs) == n, (share, len(segs))
    return segs


def analyse(csv, out, fig, pool, n_boot=20000):
    long = pd.read_csv(csv)
    labels = long[long.arm == "buy_hold"].groupby("segment").regime.first()
    starts = long[long.arm == "buy_hold"].groupby("segment").start.first()
    bear = (pd.read_parquet(pool, columns=["true_regime"]).true_regime == "bear").values
    bar_bear = pd.Series({g: bear[starts[g]:starts[g] + 720].mean() for g in labels.index})
    rng = np.random.default_rng(0)
    res = {}
    for share in SHARES:
        segs = select(labels, share)
        sub = long[long.segment.isin(segs)]
        M = {a: seed_matrix(sub, a, "final") for a in ARMS}
        r = {"bear_share": share, "n_segments": len(segs),
             # majority-labelled bull/sideways segments still hold some bear bars
             "bar_level_bear_share": float(bar_bear[segs].mean()),
             "sharpe": {a: float(M[a].mean(axis=1).mean()) for a in ARMS},
             "buy_hold": float(sub[sub.arm == "buy_hold"].sharpe.mean()),
             "timing": {a: timing_score(sub, a, "final")["timing"] for a in ARMS},
             "exposure": {a: timing_score(sub, a, "final")["exposure"] for a in ARMS},
             "pairs": {}}
        for b in ["single", "population", "reweighted"]:
            pr = bootstrap_pair(M["coevolution"], M[b], rng, n_boot)
            pr["p_reverse"] = float(1 - pr["p_hierarchical"])   # P(coevolution >= b), for ties
            r["pairs"][f"coevolution vs {b}"] = pr
        res[f"{share:.2f}"] = r
    # bear share where coevolution - single changes sign (linear interpolation)
    d = [res[f"{s:.2f}"]["pairs"]["coevolution vs single"]["mean_diff"] for s in SHARES]
    cross = [SHARES[i] - d[i] * (SHARES[i + 1] - SHARES[i]) / (d[i + 1] - d[i])
             for i in range(len(d) - 1) if np.sign(d[i]) != np.sign(d[i + 1])]
    res["sign_change_vs_single"] = cross
    json.dump(res, open(out, "w"), indent=2)

    print(f"{'bear':>5s} " + " ".join(f"{a[:10]:>11s}" for a in ARMS) + f"{'B&H':>8s}   "
          "coevo - single | population | reweighted  (diff, P hier, wins)")
    for s in SHARES:
        r = res[f"{s:.2f}"]
        print(f"{s:5.2f} " + " ".join(f"{r['sharpe'][a]:+11.3f}" for a in ARMS) + f"{r['buy_hold']:+8.3f}   "
              + " | ".join(f"{p['mean_diff']:+.2f} P{p['p_hierarchical']:.3f} {p['seed_wins']}/6"
                           for p in r["pairs"].values()))
    print("coevolution - single changes sign at bear share", [round(c, 3) for c in cross])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_, ax = plt.subplots(figsize=(6.4, 4.2))
    for a in ARMS + ["buy_hold"]:
        y = [res[f"{s:.2f}"]["sharpe"][a] if a != "buy_hold" else res[f"{s:.2f}"]["buy_hold"]
             for s in SHARES]
        ax.plot(SHARES, y, marker="o", lw=2.4 if a == "coevolution" else 1.4,
                ls="--" if a == "buy_hold" else "-", label=a.replace("_", " "))
    ax.axvline(0.13, color="grey", lw=0.8, ls=":")
    ax.text(0.135, ax.get_ylim()[0], " train mix (segment labels)", color="grey", va="bottom", fontsize=8)
    ax.set_xlabel("bear share of test segments (majority true regime)")
    sec = ax.secondary_xaxis("top")
    sec.set_xticks(SHARES, [f"{res[f'{s:.2f}']['bar_level_bear_share']:.0%}" for s in SHARES], fontsize=7)
    sec.set_xlabel("bar-level bear share", fontsize=8)
    ax.set_ylabel("mean Sharpe (final population, 6 seeds)")
    ax.set_title("SYNTH_V1 shift dose-response, 200 segments per point")
    ax.legend(fontsize=8)
    fig_.tight_layout()
    fig_.savefig(fig, dpi=150)
    print("figure", fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "analyse"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv")
    ap.add_argument("--fig", default=project_path("figures/synth_dose_response.png"))
    ap.add_argument("--pool", default=project_path("data/synth_v1_dose/pool.parquet"))
    a = ap.parse_args()
    build(a.out) if a.cmd == "build" else analyse(a.csv, a.out, a.fig, a.pool)


if __name__ == "__main__":
    main()
