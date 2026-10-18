"""Pool reruns of one frozen node config that differ only in seeds (pre-registered, R18a+R19a).

Per segment, each arm's score is averaged over the runs' seed-means; the dev rule is applied once.

    python coevo_seedpool.py r18a-progress-6-seeds r19a-progress-seeds48-53 [--coin BTCUSDT]
"""
import argparse, json, os
import numpy as np
import pandas as pd
from evaluate import paired_bootstrap

BASE = "../runs/coevo7"
ap = argparse.ArgumentParser()
ap.add_argument("nodes", nargs="+")
ap.add_argument("--coin", default="BTCUSDT")
a = ap.parse_args()
ds = [pd.read_csv(f"{BASE}/{n}/{a.coin}/per_segment.csv") for n in a.nodes]
assert all((d[["fold", "seg"]].values == ds[0][["fold", "seg"]].values).all() for d in ds)
seg = ds[0].copy()
arms = ["single", "population", "coevolution"]
for col in [x + s for x in arms for s in ("", "_2x", "_expo")]:
    seg[col] = np.mean([d[col].values for d in ds], axis=0)
c = seg.coevolution.values
others = ["single", "population", "buy_hold", "ma_crossover"]
res = {"nodes": a.nodes, "coin": a.coin, **{x: float(seg[x].mean()) for x in arms + ["buy_hold", "ma_crossover"]},
       "gaps": {o: float((c - seg[o].values).mean()) for o in others},
       "p": {o: paired_bootstrap(c, seg[o].values) for o in others},
       "coevolution_2x": float(seg.coevolution_2x.mean()), "buy_hold_2x": float(seg.buy_hold_2x.mean())}
res["pass"] = bool(all(res["gaps"][o] > 0 and res["p"][o] <= 0.10 for o in others)
                   and res["coevolution_2x"] > res["buy_hold_2x"])
print(json.dumps(res, indent=1))
