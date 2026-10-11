"""Score ONE orx strategy ONCE on the sealed span under maker model M
(docs/loop/LEDGER.md, "MAKER-FEE QUESTION"). Bars >= 76000 (2026-05-01 07:00 ->
2026-08-31), trained on the preceding 40,000 bars, 4 segments of 720 bars.

    python orx_sealed.py --strategy-dir ../../../fyp-orx/ppo/src
"""

import argparse
import importlib.util
import json

import numpy as np

from carry_check import load as load_perp
from evaluate import paired_bootstrap
from metrics import sharpe_ratio
from order_flow import load
from signal_probe import COST, SEG, run_position

SEALED_START, C_M = 76000, 0.0003
EXT = "../data/features/btcusdt_features_ext.parquet"


def run_perp(perp, paid, target, cost):
    """Same as the maker orx_eval.run_perp."""
    r = perp[1:] / perp[:-1] - 1
    pos = target[:-1]
    change = np.abs(np.diff(np.concatenate([[0.0], pos])))
    return pos * r - pos * paid[:-1] - change * cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy-dir", required=True)
    ap.add_argument("--out", default="../runs/walkforward/orx_sealed.json")
    ap.add_argument("--features", default=EXT)
    ap.add_argument("--start", type=int, default=SEALED_START, help="first scored bar")
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("orx_strategy", f"{args.strategy_dir}/orx_strategy.py")
    st = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(st)

    df, pf = load(args.features), load_perp(args.features)
    start = args.start
    n = len(df)
    assert np.allclose(pf["spot"].values, df["close"].values), "perp frame misaligned"
    fold = {"fold": "sealed", "train_start": start - 40000, "train_end": start,
            "test_start": start, "test_end": n}
    pos = np.clip(np.asarray(st.positions(df, fold), dtype=float), 0, 1)
    rows = []
    for s in range(0, n - start - SEG + 1, SEG):
        sl = slice(start + s, start + s + SEG)
        w, p, fd, c = pos[s:s + SEG], pf.perp.values[sl], pf.paid.values[sl], df.close.values[sl]
        assert np.isfinite(p).all()
        rows.append({"start": str(df.open_time.iloc[sl.start]), "move": float(np.log(c[-1] / c[0])),
                     "s1": sharpe_ratio(run_perp(p, fd, w, C_M)), "s2": sharpe_ratio(run_perp(p, fd, w, 2 * C_M)),
                     "bh": sharpe_ratio(run_position(c, np.ones(SEG), COST)),
                     "perp_bh": sharpe_ratio(run_perp(p, fd, np.ones(SEG), C_M)), "expo": float(w.mean())})
    for r in rows:
        print(f"  {r['start'][:10]}  move {r['move']:+.3f}  strategy {r['s1']:+.3f} (2x {r['s2']:+.3f})  "
              f"spot B&H {r['bh']:+.3f}  perp B&H {r['perp_bh']:+.3f}  expo {r['expo']:.2f}")
    col = lambda k: np.array([r[k] for r in rows])
    res = {"strategy": st.__doc__.strip().splitlines()[0], "n_segments": len(rows),
           "sharpe": float(col("s1").mean()), "sharpe_2x": float(col("s2").mean()),
           "buy_hold": float(col("bh").mean()), "perp_buy_hold": float(col("perp_bh").mean()),
           "wins": int((col("s1") > col("bh")).sum()), "p_le_bh": paired_bootstrap(col("s1"), col("bh")),
           "segments": rows}
    res["sealed_pass"] = res["sharpe"] > res["buy_hold"]
    print(f"SEALED: {res['strategy']}\n  Sharpe {res['sharpe']:+.3f} vs spot B&H {res['buy_hold']:+.3f} "
          f"(wins {res['wins']}/{len(rows)}, P {res['p_le_bh']:.3f})  2x {res['sharpe_2x']:+.3f}  "
          f"perp B&H {res['perp_buy_hold']:+.3f}  {'PASS' if res['sealed_pass'] else 'FAIL'}")
    json.dump(res, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
