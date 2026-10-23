"""Exposure floor: portfolio = f * B&H core + (1-f) * agent, without retraining.

From the repo root:
    MULT=1 python ppo/src/coevo_floor.py <node>

Writes <node>/BTCUSDT/floor[_x2].csv; MULT is the cost multiplier.
"""

import argparse
from functools import partial
import json
import multiprocessing as mp
import os

FLOORS = (0.0, 0.25, 0.5, 0.75)


def bh_values(seg, mult):
    import coevo_eval as E
    from baselines import buy_and_hold
    return buy_and_hold(seg, 10000.0, E.SPOT_FEE * mult, E.SPOT_SLIP * mult)


def job(j, mult):
    import numpy as np
    import coevo_eval as E
    k, path = j
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from trading_env import TradingEnv
    from metrics import summarize
    _, test = E.DATA[k]
    segs = [(s, s + E.SEG) for s in range(0, len(test) - E.SEG + 1, E.SEG)]
    d = os.path.join(os.path.dirname(path), "final_traders")
    model = E.SoftVote([PPO.load(os.path.join(d, f), device="cpu") for f in sorted(os.listdir(d)) if f.endswith(".zip")])
    eng = E.CFG.get("engine", {})
    out = []
    for i, (s, e) in enumerate(segs):
        env = TradingEnv(df=test, feature_columns=E.FEATS, segments=segs, initial_balance=10000.0,
                         trading_fee=E.AGENT_FEE * mult, slippage=E.AGENT_SLIP * mult, trade_penalty=0.0,
                         funding_col="funding_paid", decision_every=eng.get("decision_every", 1),
                         start_long=eng.get("start_long", False), target_position=eng.get("target_position", False))
        obs, _ = env.reset(seed=0, options={"segment": i})
        v, done = [10000.0], False
        while not done:
            obs, _, done, _, info = env.step(model.predict(obs)[0])
            v.extend(info["bar_values"])
        bh = bh_values(test.iloc[s:e].reset_index(drop=True), mult)
        n = min(len(v), len(bh))
        v, bh = np.asarray(v[:n]), np.asarray(bh[:n])
        out.append([summarize(f * bh + (1 - f) * v)["sharpe"] for f in FLOORS])
    return j, out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("node")
    args = ap.parse_args(argv)

    import numpy as np
    import pandas as pd
    import coevo_eval as E

    mult = float(os.environ.get("MULT", "1"))
    root = f"{E.OUT_BASE}/{args.node}/BTCUSDT"
    with open(f"{root}/contract.json") as contract:
        E.CFG = json.load(contract)["config"]
    seeds = E.CFG["seeds"]
    assert E.CFG.get("score") == "vote"   # job loads final_traders as SoftVote, like eval_job

    E.DATA, E.FEATS, _ = E.prepare(E.SYM, E.CFG)
    jobs = [(k, f"{root}/fold{k}/{a}_seed{s}/best_trader.zip") for a in E.ARMS for k in sorted(E.DATA) for s in seeds]
    with mp.get_context("fork").Pool(11) as p:
        res = dict(p.map(partial(job, mult=mult), jobs))
    rows = []
    for k in sorted(E.DATA):
        b = E.baselines(E.DATA[k][1], mult)
        for i in range(len(b)):
            r = {"fold": k, "seg": i, "buy_hold": b[i]["buy_hold"], "ma_crossover": b[i]["ma_crossover"]}
            for a in E.ARMS:
                sh = np.mean([res[(k, f"{root}/fold{k}/{a}_seed{s}/best_trader.zip")][i] for s in seeds], axis=0)
                for f, x in zip(FLOORS, sh):
                    r[f"{a}_f{f:g}"] = float(x)
            rows.append(r)
    pd.DataFrame(rows).to_csv(f"{root}/floor{'' if mult == 1 else f'_x{mult:g}'}.csv", index=False)


if __name__ == "__main__":
    main()
