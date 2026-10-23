"""Pool reruns of one frozen node config that differ only in seeds.

Per segment, each arm's score is averaged over the runs' seed-means; the dev rule is applied once.

    python coevo_seedpool.py r21a-pure-alp r22a-pure-alp-seeds48-53
"""
import argparse
import json

from project_paths import project_path

BASE = project_path("runs/coevo7")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("nodes", nargs="+")
    a = ap.parse_args(argv)

    import numpy as np
    import pandas as pd
    from evaluate import paired_bootstrap

    ds = [pd.read_csv(f"{BASE}/{n}/BTCUSDT/per_segment.csv") for n in a.nodes]
    assert all((d[["fold", "seg"]].values == ds[0][["fold", "seg"]].values).all() for d in ds)
    seg = ds[0].copy()
    arms = ["single", "population", "coevolution"]
    for col in [x + s for x in arms for s in ("", "_2x", "_expo")]:
        seg[col] = np.mean([d[col].values for d in ds], axis=0)
    c = seg.coevolution.values
    others = ["single", "population", "buy_hold", "ma_crossover"]
    res = {"nodes": a.nodes, **{x: float(seg[x].mean()) for x in arms + ["buy_hold", "ma_crossover"]},
           "gaps": {o: float((c - seg[o].values).mean()) for o in others},
           "p": {o: paired_bootstrap(c, seg[o].values) for o in others},
           "coevolution_2x": float(seg.coevolution_2x.mean()), "buy_hold_2x": float(seg.buy_hold_2x.mean())}
    res["pass"] = bool(all(res["gaps"][o] > 0 and res["p"][o] <= 0.10 for o in others)
                       and res["coevolution_2x"] > res["buy_hold_2x"])
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
