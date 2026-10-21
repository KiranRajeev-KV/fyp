"""V7 coevolution search: fixed run contract (docs/loop/LEDGER.md, "COEVO V7 GOAL").

Trains the three engine arms (single, population, coevolution) on each of the
six BTC walk-forward folds with the node's settings (coevo_node.py), then
scores every arm's frozen best trader (mean over seeds) on the 48 dev test
segments against spot buy-and-hold and MA crossover. Never reads BTC bars
from 2026-05-01 07:00 on (sealed / spent).

FIXED here, whatever the node says: the cost model (agent = perp maker 0.02%
fee + 0.01% slippage per side + funding while long; B&H = spot 0.1% + 0.05%;
MA crossover = the agent's fee/slippage, no funding), the folds, the
segments, the selection rule (best_trader per seed), the pass rule.

    python coevo_eval.py                       # BTC dev (the orx run command)
    python coevo_eval.py --coin ETHUSDT        # confirmation, run ONCE on a frozen node
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time

import numpy as np
import pandas as pd

AGENT_FEE, AGENT_SLIP = 0.0002, 0.0001
SPOT_FEE, SPOT_SLIP = 0.001, 0.0005
SEG, VAL_SPAN, POOL_STRIDE = 720, 4000, 168
ARMS = ["single", "population", "coevolution"]
OUT_BASE = "/home/nandu/uni/fyp/fyp/ppo/runs/coevo7"


def rhat_feature(df, cols, folds_win, horizon):
    """Ridge forecast of the next `horizon`-bar log return, no lookahead.

    Train rows get 5-block cross-fitted predictions (purged by `horizon` at
    block edges) so PPO never sees an in-sample fit; test rows get the
    prediction of a fit on the whole train window (labels purged at its end).
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    y_all = np.log(df.close.shift(-horizon) / df.close).values
    out = {}
    for w in folds_win:
        tr = np.arange(w["train_start"], w["train_end"])
        X = df[cols].values
        def fit(idx):
            idx = idx[np.isfinite(y_all[idx])]
            sc = StandardScaler().fit(X[idx])
            Z = np.nan_to_num(sc.transform(X[idx]))
            return sc, Ridge(alpha=1.0).fit(Z, y_all[idx])
        def pred(m, idx):
            return m[1].predict(np.nan_to_num(m[0].transform(X[idx])))
        r = np.full(len(df), np.nan)
        blocks = np.array_split(tr, 5)
        for b in blocks:
            keep = tr[(tr < b[0] - horizon) | (tr > b[-1] + horizon)]
            r[b] = pred(fit(keep), b)
        r[w["test_start"]:w["test_end"]] = pred(fit(tr[:-horizon]), np.arange(w["test_start"], w["test_end"]))
        out[w["fold"]] = r
    return out


def prepare(sym, cfg):
    from coin_data import FLOW, POS, load_coin, fold_windows
    from build_features import FEATURE_COLUMNS
    df = load_coin(sym)
    wins = fold_windows(df)
    cols = cfg.get("features") or list(FEATURE_COLUMNS) + (FLOW + POS if cfg.get("flow_pos", True) else [])
    # rhat: one horizon (feature "rhat") or a list of horizons (features "rhat<h>")
    hs = cfg.get("rhat") or []
    names = {hs: "rhat"} if isinstance(hs, int) else {h: f"rhat{h}" for h in hs}
    rh = {n: rhat_feature(df, list(FEATURE_COLUMNS) + FLOW + POS, wins, h) for h, n in names.items()}
    feats = cols + list(rh)
    data = {}
    for w in wins:
        d = df.copy()
        for n, r in rh.items():
            d[n] = r[w["fold"]]
        train = d.iloc[w["train_start"]:w["train_end"]].reset_index(drop=True)
        test = d.iloc[w["test_start"]:w["test_end"]].reset_index(drop=True)
        mu, sd = train[feats].mean(), train[feats].std().replace(0, 1)
        for x in (train, test):
            x.loc[:, feats] = ((x[feats] - mu) / sd).clip(-10, 10).fillna(0.0)
        data[w["fold"]] = (train, test)
    return data, feats, wins


def merged_pool(frames):
    """One segment pool over several coins' train frames; no segment crosses a coin boundary."""
    import dataclasses
    from segments import SegmentPool, build_segment_pool
    dfs, segs, off = [], [], 0
    for f in frames:
        p = build_segment_pool(f.reset_index(drop=True), segment_length=SEG, stride=POOL_STRIDE)
        segs += [dataclasses.replace(sg, start=sg.start + off, end=sg.end + off) for sg in p.segments]
        dfs.append(p.df)
        off += len(p.df)
    return SegmentPool(df=pd.concat(dfs, ignore_index=True), segments=segs)


def engine_config(cfg, arm, seed, sym):
    from coevo.engine import EngineConfig
    kw = dict(cfg.get("engine", {}))
    kw.update(cfg.get("arm_overrides", {}).get(arm, {}))
    kw.update(mode=arm, seed=seed, trading_fee=AGENT_FEE, slippage=AGENT_SLIP,
              funding_col="funding_paid", resume=True)
    return EngineConfig(**kw)


def train_job(job):
    fold, arm, seed, out_dir = job
    import torch
    torch.set_num_threads(1)
    from coevo.engine import CoevolutionEngine
    train, _ = DATA[fold]
    if os.path.exists(os.path.join(out_dir, "best_trader.zip")):
        return job, 0.0
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    val = train.iloc[-VAL_SPAN:].reset_index(drop=True)
    # aux coins (node "aux_coins"): their bars from the same calendar train window, val span cut too
    pool = merged_pool([train.iloc[:-VAL_SPAN]] + [AUX[c][fold][0].iloc[:-VAL_SPAN] for c in AUX])
    t0 = time.time()
    import contextlib
    with open(os.path.join(os.path.dirname(out_dir), f"{os.path.basename(out_dir)}.log"), "w") as log, \
            contextlib.redirect_stdout(log):
        CoevolutionEngine(engine_config(CFG, arm, seed, SYM), pool, FEATS, val, out_dir).run()
    return job, time.time() - t0


class SoftVote:
    """An arm's final population as one policy: argmax of the mean action probabilities."""

    def __init__(self, models):
        self.models = models

    def predict(self, obs, deterministic=True):
        import torch
        with torch.no_grad():
            p = [m.policy.get_distribution(m.policy.obs_to_tensor(obs)[0]).distribution.probs.numpy()[0]
                 for m in self.models]
        return int(np.mean(p, axis=0).argmax()), None


def eval_job(job):
    coin, fold, path, mult = job
    import torch
    torch.set_num_threads(1)
    from types import SimpleNamespace
    from stable_baselines3 import PPO
    from coevo.trader_population import evaluate_on_segments
    _, test = TEST[coin][fold]
    segs = [(s, s + SEG) for s in range(0, len(test) - SEG + 1, SEG)]
    kw = {"initial_balance": 10000.0, "trading_fee": AGENT_FEE * mult, "slippage": AGENT_SLIP * mult,
          "trade_penalty": 0.0, "funding_col": "funding_paid",
          "decision_every": CFG.get("engine", {}).get("decision_every", 1),
          "start_long": CFG.get("engine", {}).get("start_long", False),
          "target_position": CFG.get("engine", {}).get("target_position", False)}
    if CFG.get("score") == "vote":       # the seed's final population votes (single: 1 trader)
        d = os.path.join(os.path.dirname(path), "final_traders")
        model = SoftVote([PPO.load(os.path.join(d, f), device="cpu") for f in sorted(os.listdir(d)) if f.endswith(".zip")])
    else:
        model = PPO.load(path, device="cpu")
    _, rows = evaluate_on_segments(SimpleNamespace(model=model), test, FEATS, segs, kw)
    return job, [(r["sharpe"], r["exposure"]) for r in rows]


def baselines(test, mult):
    from baselines import buy_and_hold, ma_crossover
    from metrics import summarize
    out = []
    for s in range(0, len(test) - SEG + 1, SEG):
        seg = test.iloc[s:s + SEG].reset_index(drop=True)
        out.append({"buy_hold": summarize(buy_and_hold(seg, 10000.0, SPOT_FEE * mult, SPOT_SLIP * mult))["sharpe"],
                    "ma_crossover": summarize(ma_crossover(seg, 10000.0, AGENT_FEE * mult,
                                                           slippage=AGENT_SLIP * mult))["sharpe"],
                    "move": float(np.log(seg.close.iloc[-1] / seg.close.iloc[0]))})
    return out


def main():
    global DATA, FEATS, CFG, SYM, AUX, TEST
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="BTCUSDT")
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--folds", type=int, nargs="*", default=None, help="smoke tests only")
    ap.add_argument("--test-coins", nargs="*", default=None,
                    help="zero-shot test coins (V8); default: the node's test_coins, else --coin")
    args = ap.parse_args()
    import coevo_node
    CFG, SYM = coevo_node.CONFIG, args.coin
    from evaluate import paired_bootstrap

    DATA, FEATS, wins = prepare(SYM, CFG)
    AUX = {c: prepare(c, CFG)[0] for c in CFG.get("aux_coins", []) if c != SYM}
    test_coins = args.test_coins or CFG.get("test_coins") or [SYM]
    tag = "" if test_coins == [SYM] else "_" + "_".join(test_coins)   # zero-shot results never overwrite the dev ones
    folds = args.folds if args.folds is not None else [w["fold"] for w in wins]
    out_root = os.path.join(OUT_BASE, CFG["name"], SYM)
    os.makedirs(out_root, exist_ok=True)
    json.dump({"config": CFG, "features": FEATS, "folds": wins}, open(os.path.join(out_root, "contract.json"), "w"),
              indent=2, default=str)
    seeds = CFG["seeds"]
    jobs = [(k, arm, s, os.path.join(out_root, f"fold{k}", f"{arm}_seed{s}"))
            for arm in ["coevolution", "population", "single"] for k in folds for s in seeds]   # slow first
    for j in jobs:
        os.makedirs(os.path.dirname(j[3]), exist_ok=True)
    ctx = mp.get_context("fork")
    t0 = time.time()
    with ctx.Pool(args.workers) as pool:
        for (k, arm, s, _), sec in pool.imap_unordered(train_job, jobs):
            print(f"  trained fold {k} {arm} seed {s} in {sec / 60:.1f} min  [{(time.time() - t0) / 60:.0f} min]",
                  flush=True)
    # test coins load after training (a training run needs no test data); fresh workers see TEST
    TEST = {c: DATA if c == SYM else prepare(c, CFG)[0] for c in test_coins}
    with ctx.Pool(args.workers) as pool:
        ev = [(c, k, os.path.join(out_root, f"fold{k}", f"{arm}_seed{s}", "best_trader.zip"), m)
              for c in test_coins for arm in ARMS for k in folds for s in seeds for m in (1.0, 2.0)]
        res = dict(pool.map(eval_job, ev))

    rows = []
    for c, k in [(c, k) for c in test_coins for k in folds]:
        b1, b2 = baselines(TEST[c][k][1], 1.0), baselines(TEST[c][k][1], 2.0)
        for i in range(len(b1)):
            row = {"coin": c, "fold": k, "seg": i, "move": b1[i]["move"], "buy_hold": b1[i]["buy_hold"],
                   "ma_crossover": b1[i]["ma_crossover"], "buy_hold_2x": b2[i]["buy_hold"],
                   "ma_crossover_2x": b2[i]["ma_crossover"]}
            for arm in ARMS:
                for m, suf in ((1.0, ""), (2.0, "_2x")):
                    vals = [res[(c, k, os.path.join(out_root, f"fold{k}", f"{arm}_seed{s}", "best_trader.zip"), m)][i]
                            for s in seeds]
                    row[arm + suf] = float(np.mean([v[0] for v in vals]))
                    if m == 1.0:
                        row[arm + "_expo"] = float(np.mean([v[1] for v in vals]))
            rows.append(row)
    seg = pd.DataFrame(rows)
    seg.to_csv(os.path.join(out_root, f"per_segment{tag}.csv"), index=False)

    c = seg.coevolution.values
    others = ["single", "population", "buy_hold", "ma_crossover"]
    res = {"coin": "+".join(test_coins), "train_coin": SYM, "node": CFG["name"], "n_segments": len(seg),
           **{a: float(seg[a].mean()) for a in ARMS + ["buy_hold", "ma_crossover"]},
           "gaps": {o: float((c - seg[o].values).mean()) for o in others},
           "p": {o: paired_bootstrap(c, seg[o].values) for o in others},
           "wins": {o: int((c > seg[o].values).sum()) for o in others},
           "coevolution_2x": float(seg.coevolution_2x.mean()), "buy_hold_2x": float(seg.buy_hold_2x.mean()),
           "exposure": {a: float(seg[a + "_expo"].mean()) for a in ARMS},
           "timing": {a: float(seg[a + "_expo"][seg.move > 0].mean() - seg[a + "_expo"][seg.move < 0].mean())
                      for a in ARMS},
           "per_fold_gap_bh": seg.assign(g=c - seg.buy_hold).groupby("fold").g.mean().round(3).to_dict(),
           "per_coin_gap_pop": seg.assign(g=c - seg.population).groupby("coin").g.mean().round(3).to_dict(),
           "minutes": round((time.time() - t0) / 60, 1)}
    res["pass"] = bool(all(res["gaps"][o] > 0 and res["p"][o] <= 0.10 for o in others)
                       and res["coevolution_2x"] > res["buy_hold_2x"])
    json.dump(res, open(os.path.join(out_root, f"result{tag}.json"), "w"), indent=2)
    print(f"node {CFG['name']} on {res['coin']}: " + "  ".join(f"{a} {res[a]:+.3f}" for a in ARMS + ["buy_hold", "ma_crossover"]))
    print("  coevolution gap / P: " + "  ".join(f"{o} {res['gaps'][o]:+.3f}/{res['p'][o]:.3f}" for o in others)
          + f"  2x {res['coevolution_2x']:+.3f} vs B&H {res['buy_hold_2x']:+.3f}"
          + f"  expo {res['exposure']}  {'PASS' if res['pass'] else '-'}")
    print("ORX_RESULT " + json.dumps(res))


if __name__ == "__main__":
    sys.exit(main())
