# Exposure floor: portfolio = f * B&H core + (1-f) * agent, for frozen BTC models (no retraining).
#   MULT=1 python coevo_floor.py <node>  ->  <node>/BTCUSDT/floor[_x2].csv   (MULT = cost multiplier)
import sys, os, json, multiprocessing as mp, numpy as np, pandas as pd
import coevo_eval as E
FLOORS = (0.0, 0.25, 0.5, 0.75)
node = sys.argv[1]
M = float(os.environ.get("MULT", "1"))
root = f"{E.OUT_BASE}/{node}/BTCUSDT"
E.CFG = json.load(open(f"{root}/contract.json"))["config"]
seeds = E.CFG["seeds"]
assert E.CFG.get("score") == "vote"   # rollout below loads final_traders as SoftVote, like eval_job


def bh_values(seg):
    from baselines import buy_and_hold
    return buy_and_hold(seg, 10000.0, E.SPOT_FEE * M, E.SPOT_SLIP * M)


def job(j):
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
                         trading_fee=E.AGENT_FEE * M, slippage=E.AGENT_SLIP * M, trade_penalty=0.0,
                         funding_col="funding_paid", decision_every=eng.get("decision_every", 1),
                         start_long=eng.get("start_long", False), target_position=eng.get("target_position", False))
        obs, _ = env.reset(seed=0, options={"segment": i})
        v, done = [10000.0], False
        while not done:
            obs, _, done, _, info = env.step(model.predict(obs)[0])
            v.extend(info["bar_values"])
        bh = bh_values(test.iloc[s:e].reset_index(drop=True))
        n = min(len(v), len(bh))
        v, bh = np.asarray(v[:n]), np.asarray(bh[:n])
        out.append([summarize(f * bh + (1 - f) * v)["sharpe"] for f in FLOORS])
    return j, out


E.DATA, E.FEATS, _ = E.prepare(E.SYM, E.CFG)
jobs = [(k, f"{root}/fold{k}/{a}_seed{s}/best_trader.zip") for a in E.ARMS for k in sorted(E.DATA) for s in seeds]
with mp.get_context("fork").Pool(11) as p:
    res = dict(p.map(job, jobs))
rows = []
for k in sorted(E.DATA):
    b = E.baselines(E.DATA[k][1], M)
    for i in range(len(b)):
        r = {"fold": k, "seg": i, "buy_hold": b[i]["buy_hold"], "ma_crossover": b[i]["ma_crossover"]}
        for a in E.ARMS:
            sh = np.mean([res[(k, f"{root}/fold{k}/{a}_seed{s}/best_trader.zip")][i] for s in seeds], axis=0)
            for f, x in zip(FLOORS, sh):
                r[f"{a}_f{f:g}"] = float(x)
        rows.append(r)
pd.DataFrame(rows).to_csv(f"{root}/floor{'' if M == 1 else f'_x{M:g}'}.csv", index=False)
