"""Historical ORX development contract (local ledger label: GOAL MODE).

Scores orx_strategy.positions() on the 48 development segments against
buy-and-hold. Data loaders can read full files; scoring excludes bars from
SEALED_START onward. The May–September 2026 checks are spent historical data,
not fresh holdouts. ORX_RESULT is the retained machine-readable output prefix.

From the repo root:
    python ppo/src/orx_eval.py
"""

import json

import numpy as np

import orx_strategy
from evaluate import paired_bootstrap
from metrics import sharpe_ratio
from order_flow import load
from signal_probe import COST, SEG, run_position
from project_paths import project_path

SEALED_START = 76000          # 2026-05-01 07:00 UTC: first bar of the sealed holdout


def main():
    df = load().iloc[:SEALED_START]
    folds = json.load(open(project_path("runs/walkforward/folds.json")))["folds"]
    assert max(f["test_end"] for f in folds) <= SEALED_START
    close = df["close"].values
    rows = []
    for fold in folds:
        n = fold["test_end"] - fold["test_start"]
        pos = np.asarray(orx_strategy.positions(df.iloc[:fold["test_end"]], fold), dtype=float)
        assert pos.shape == (n,) and np.isfinite(pos).all(), f"bad positions for fold {fold['fold']}"
        pos = np.clip(pos, 0.0, 1.0)
        c = close[fold["test_start"]:fold["test_end"]]
        for s in range(0, n - SEG + 1, SEG):
            cs, w = c[s:s + SEG], pos[s:s + SEG]
            row = {"move": float(np.log(cs[-1] / cs[0])), "expo": float(w.mean()),
                   "turnover": float(np.abs(np.diff(np.concatenate([[0.0], w]))).sum())}
            for m in (0.0, 1.0, 2.0):
                row[f"s{m:g}"] = sharpe_ratio(run_position(cs, w, COST * m))
                row[f"b{m:g}"] = sharpe_ratio(run_position(cs, np.ones(SEG), COST * m))
            rows.append(row)

    col = lambda k: np.array([r[k] for r in rows])
    up, down = col("move") > 0, col("move") < 0
    p = paired_bootstrap(col("s1"), col("b1"))
    res = {"n_segments": len(rows),
           "sharpe": float(col("s1").mean()), "buy_hold": float(col("b1").mean()),
           "gap": float((col("s1") - col("b1")).mean()), "p_le_bh": p,
           "wins": int((col("s1") > col("b1")).sum()),
           "sharpe_2x": float(col("s2").mean()), "buy_hold_2x": float(col("b2").mean()),
           "sharpe_0x": float(col("s0").mean()), "buy_hold_0x": float(col("b0").mean()),
           "exposure": float(col("expo").mean()), "turnover_per_seg": float(col("turnover").mean()),
           "timing": float(col("expo")[up].mean() - col("expo")[down].mean())}
    res["dev_pass"] = bool(res["gap"] > 0 and p <= 0.10 and res["sharpe_2x"] > res["buy_hold_2x"])
    print(f"strategy: {orx_strategy.__doc__.strip().splitlines()[0]}")
    print(f"Sharpe {res['sharpe']:+.3f} vs B&H {res['buy_hold']:+.3f} (gap {res['gap']:+.3f}, P {p:.3f}, "
          f"wins {res['wins']}/{len(rows)})  2x {res['sharpe_2x']:+.3f} vs {res['buy_hold_2x']:+.3f}  "
          f"0x {res['sharpe_0x']:+.3f}  expo {res['exposure']:.2f}  turnover {res['turnover_per_seg']:.1f}/seg  "
          f"timing {res['timing']:+.3f}  {'DEV PASS' if res['dev_pass'] else '-'}")
    print("ORX_RESULT " + json.dumps(res))


if __name__ == "__main__":
    main()
