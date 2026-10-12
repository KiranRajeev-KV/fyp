"""V7 confirmation: pool a frozen node's per-segment results over coins, apply the pass rule ONCE.

    python coevo_pool.py <node name> ETHUSDT SOLUSDT
"""

import json
import os
import sys

import numpy as np
import pandas as pd

from coevo_eval import OUT_BASE
from evaluate import paired_bootstrap

OTHERS = ["single", "population", "buy_hold", "ma_crossover"]


def verdict(seg):
    c = seg.coevolution.values
    res = {"n_segments": len(seg),
           **{a: float(seg[a].mean()) for a in ["coevolution"] + OTHERS},
           "gaps": {o: float((c - seg[o].values).mean()) for o in OTHERS},
           "p": {o: paired_bootstrap(c, seg[o].values) for o in OTHERS},
           "coevolution_2x": float(seg.coevolution_2x.mean()), "buy_hold_2x": float(seg.buy_hold_2x.mean())}
    res["pass"] = bool(all(res["gaps"][o] > 0 and res["p"][o] <= 0.10 for o in OTHERS)
                       and res["coevolution_2x"] > res["buy_hold_2x"])
    return res


if __name__ == "__main__":
    node, coins = sys.argv[1], sys.argv[2:]
    assert coins, "name the coins"
    seg = pd.concat([pd.read_csv(os.path.join(OUT_BASE, node, c, "per_segment.csv")).assign(coin=c) for c in coins])
    res = {"node": node, "coins": coins, **verdict(seg)}
    out = os.path.join(OUT_BASE, node, "confirm_" + "_".join(coins) + ".json")
    json.dump(res, open(out, "w"), indent=2)
    print(json.dumps(res, indent=1))
    print("CONFIRMED" if res["pass"] else "NOT CONFIRMED", out)
