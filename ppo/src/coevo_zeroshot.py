# Zero-shot probe: BTC-trained node models evaluated on other coins' test windows (same folds).
import sys, os, json, multiprocessing as mp, numpy as np, pandas as pd
os.chdir("/home/nandu/uni/fyp/fyp-coevo/ppo/src"); sys.path.insert(0, ".")
import coevo_eval as E
from evaluate import paired_bootstrap
node, coins = sys.argv[1], sys.argv[2:]
M = float(os.environ.get("MULT", "1"))
root = f"{E.OUT_BASE}/{node}/BTCUSDT"
E.CFG = json.load(open(f"{root}/contract.json"))["config"]
seeds = E.CFG["seeds"]
rows = []
for coin in coins:
    E.DATA, E.FEATS, wins = E.prepare(coin, E.CFG)
    folds = sorted(E.DATA)
    E.TEST = {coin: E.DATA}
    jobs = [(coin, k, f"{root}/fold{k}/{a}_seed{s}/best_trader.zip", M) for a in E.ARMS for k in folds for s in seeds]
    with mp.get_context("fork").Pool(11) as p:
        res = dict(p.map(E.eval_job, jobs))
    for k in folds:
        b = E.baselines(E.DATA[k][1], M)
        for i in range(len(b)):
            r = {"coin": coin, "fold": k, "seg": i, "buy_hold": b[i]["buy_hold"], "ma_crossover": b[i]["ma_crossover"]}
            for a in E.ARMS:
                r[a] = float(np.mean([res[(coin, k, f"{root}/fold{k}/{a}_seed{s}/best_trader.zip", M)][i][0] for s in seeds]))
            rows.append(r)
    print(coin, "done", flush=True)
d = pd.DataFrame(rows); d.to_csv(f"{root}/zeroshot_{"_".join(coins)}{"" if M == 1 else f"_x{M:g}"}.csv", index=False)
print(node, "n", len(d), {a: round(d[a].mean(), 3) for a in E.ARMS + ["buy_hold", "ma_crossover"]})
for o in ["single", "population", "buy_hold", "ma_crossover"]:
    print(f"  coevo vs {o}: gap {(d.coevolution - d[o]).mean():+.3f} P {paired_bootstrap(d.coevolution.values, d[o].values):.3f}")
print(d.groupby("coin")[E.ARMS + ["buy_hold"]].mean().round(2))
