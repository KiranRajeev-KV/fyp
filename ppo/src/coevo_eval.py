"""Final BTC evaluation on six rolling development folds (48 scoring segments).

The node in coevo_node.py selects features, engine overrides, seeds, and policy
scoring. With score='vote', each seed's final traders vote by their mean action
probabilities; otherwise its saved best_trader is scored. Segment scores are
averaged over seeds before paired segment bootstrap comparisons.

Fixed costs: agents use perp maker fee 0.02% + slippage 0.01% per side and
funding while long; spot buy-and-hold uses 0.1% + 0.05%; MA crossover uses the
agent fee/slippage without funding. The fold geometry, scoring segments, and
pass rule are fixed independently of node settings. Loaders may read full
files, but training/scoring excludes BTC bars from 2026-05-01 07:00 UTC onward.
Historical checks have spent May–September 2026; these are development folds,
not a new prospective test. See ppo/README.md for the four-run headline matrix.

From the repo root:
    python ppo/src/coevo_eval.py
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
from project_paths import project_path

AGENT_FEE, AGENT_SLIP = 0.0002, 0.0001
SPOT_FEE, SPOT_SLIP = 0.001, 0.0005
SEG, VAL_SPAN, POOL_STRIDE = 720, 4000, 168
ARMS = ["single", "population", "coevolution"]
SYM = "BTCUSDT"
OUT_BASE = project_path("runs/coevo7")


def rhat_feature(df, cols, folds_win, horizon):
    """Ridge forecast with block-cross-fitted training inputs and purged test fits.

    Train rows get 5-block cross-fitted predictions (purged by `horizon` at
    block edges); the other training blocks include later rows, so training
    predictions are out of sample but not strictly chronological. Test rows
    use a fit on the whole train window with its last `horizon` labels removed.
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


def engine_config(cfg, arm, seed):
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
    from segments import build_segment_pool
    pool = build_segment_pool(train.iloc[:-VAL_SPAN].reset_index(drop=True), segment_length=SEG, stride=POOL_STRIDE)
    t0 = time.time()
    import contextlib
    with open(os.path.join(os.path.dirname(out_dir), f"{os.path.basename(out_dir)}.log"), "w") as log, \
            contextlib.redirect_stdout(log):
        CoevolutionEngine(engine_config(CFG, arm, seed), pool, FEATS, val, out_dir).run()
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
    fold, path, mult = job
    import torch
    torch.set_num_threads(1)
    from types import SimpleNamespace
    from stable_baselines3 import PPO
    from coevo.trader_population import evaluate_on_segments
    _, test = DATA[fold]
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
    global DATA, FEATS, CFG
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--folds", type=int, nargs="*", default=None, help="smoke tests only")
    args = ap.parse_args()
    import coevo_node
    CFG = coevo_node.CONFIG
    from evaluate import paired_bootstrap

    DATA, FEATS, wins = prepare(SYM, CFG)
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
    with ctx.Pool(args.workers) as pool:
        ev = [(k, os.path.join(out_root, f"fold{k}", f"{arm}_seed{s}", "best_trader.zip"), m)
              for arm in ARMS for k in folds for s in seeds for m in (1.0, 2.0)]
        res = dict(pool.map(eval_job, ev))

    rows = []
    for k in folds:
        b1, b2 = baselines(DATA[k][1], 1.0), baselines(DATA[k][1], 2.0)
        for i in range(len(b1)):
            row = {"fold": k, "seg": i, "move": b1[i]["move"], "buy_hold": b1[i]["buy_hold"],
                   "ma_crossover": b1[i]["ma_crossover"], "buy_hold_2x": b2[i]["buy_hold"],
                   "ma_crossover_2x": b2[i]["ma_crossover"]}
            for arm in ARMS:
                for m, suf in ((1.0, ""), (2.0, "_2x")):
                    vals = [res[(k, os.path.join(out_root, f"fold{k}", f"{arm}_seed{s}", "best_trader.zip"), m)][i]
                            for s in seeds]
                    row[arm + suf] = float(np.mean([v[0] for v in vals]))
                    if m == 1.0:
                        row[arm + "_expo"] = float(np.mean([v[1] for v in vals]))
            rows.append(row)
    seg = pd.DataFrame(rows)
    seg.to_csv(os.path.join(out_root, "per_segment.csv"), index=False)

    c = seg.coevolution.values
    others = ["single", "population", "buy_hold", "ma_crossover"]
    res = {"node": CFG["name"], "n_segments": len(seg),
           **{a: float(seg[a].mean()) for a in ARMS + ["buy_hold", "ma_crossover"]},
           "gaps": {o: float((c - seg[o].values).mean()) for o in others},
           "p": {o: paired_bootstrap(c, seg[o].values) for o in others},
           "wins": {o: int((c > seg[o].values).sum()) for o in others},
           "coevolution_2x": float(seg.coevolution_2x.mean()), "buy_hold_2x": float(seg.buy_hold_2x.mean()),
           "exposure": {a: float(seg[a + "_expo"].mean()) for a in ARMS},
           "timing": {a: float(seg[a + "_expo"][seg.move > 0].mean() - seg[a + "_expo"][seg.move < 0].mean())
                      for a in ARMS},
           "per_fold_gap_bh": seg.assign(g=c - seg.buy_hold).groupby("fold").g.mean().round(3).to_dict(),
           "minutes": round((time.time() - t0) / 60, 1)}
    res["pass"] = bool(all(res["gaps"][o] > 0 and res["p"][o] <= 0.10 for o in others)
                       and res["coevolution_2x"] > res["buy_hold_2x"])
    json.dump(res, open(os.path.join(out_root, "result.json"), "w"), indent=2)
    print(f"node {CFG['name']}: " + "  ".join(f"{a} {res[a]:+.3f}" for a in ARMS + ["buy_hold", "ma_crossover"]))
    print("  coevolution gap / P: " + "  ".join(f"{o} {res['gaps'][o]:+.3f}/{res['p'][o]:.3f}" for o in others)
          + f"  2x {res['coevolution_2x']:+.3f} vs B&H {res['buy_hold_2x']:+.3f}"
          + f"  expo {res['exposure']}  {'PASS' if res['pass'] else '-'}")
    print("ORX_RESULT " + json.dumps(res))


if __name__ == "__main__":
    sys.exit(main())
