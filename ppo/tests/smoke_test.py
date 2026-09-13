"""End-to-end smoke test on synthetic data.

Generates a small synthetic OHLCV series (GBM with regime shifts), builds
the segment pool, runs a tiny run of each engine mode for 2 generations,
and runs the evaluation harness pieces. Verifies the machinery executes —
it says nothing about trading performance.

Run (from ppo/tests, with the ppo/.venv active):
    python smoke_test.py
"""

import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from segments import build_segment_pool, walk_forward_folds  # noqa: E402
from regimes import add_regime_column  # noqa: E402
from trading_env import TradingEnv  # noqa: E402
from baselines import buy_and_hold, ma_crossover  # noqa: E402
from metrics import summarize  # noqa: E402
from coevo.engine import CoevolutionEngine, EngineConfig  # noqa: E402

FEATURES = ["log_return", "vol_24", "mom_24", "rsi_proxy"]


def synthetic_df(n=6000, seed=0):
    """GBM with alternating drift so bull/bear/sideways regimes all appear."""
    rng = np.random.default_rng(seed)
    drift = np.concatenate([
        np.full(n // 3, 0.0004),
        np.full(n // 3, -0.0004),
        np.full(n - 2 * (n // 3), 0.0),
    ])
    log_ret = drift + rng.normal(0, 0.004, size=n)
    close = 20000 * np.exp(np.cumsum(log_ret))

    df = pd.DataFrame({
        "open": close * (1 + rng.normal(0, 0.0005, n)),
        "high": close * (1 + abs(rng.normal(0, 0.001, n))),
        "low": close * (1 - abs(rng.normal(0, 0.001, n))),
        "close": close,
        "volume": rng.uniform(100, 1000, n),
    })
    df["log_return"] = np.log(df["close"]).diff().fillna(0)
    df["vol_24"] = df["log_return"].rolling(24).std().fillna(0)
    df["mom_24"] = df["close"].pct_change(24).fillna(0)
    df["rsi_proxy"] = df["log_return"].rolling(14).mean().fillna(0)
    return df


def main():
    df = synthetic_df()
    train_df = df.iloc[:4500].reset_index(drop=True)
    val_df = df.iloc[4500:].reset_index(drop=True)

    # --- regimes ---
    labeled = add_regime_column(train_df, window=240, threshold=0.05)
    counts = labeled["regime"].value_counts()
    assert len(counts) >= 2, f"expected multiple regimes, got {counts.to_dict()}"
    print("regimes:", counts.to_dict())

    # --- walk-forward folds ---
    folds = walk_forward_folds(len(df), train_span=3000, test_span=1000)
    assert len(folds) >= 2
    print(f"walk-forward folds: {len(folds)}")

    # --- segment pool ---
    pool = build_segment_pool(train_df, segment_length=240, stride=120,
                              regime_window=240, regime_threshold=0.05)
    assert len(pool) > 10
    assert pool.descriptor_matrix.shape == (len(pool), 5)
    print(f"segment pool: {len(pool)} segments, regimes {pool.regimes()}")

    # --- env: slippage + segment parameterization ---
    env = TradingEnv(train_df, FEATURES, segments=[(0, 240), (240, 480)],
                     slippage=0.001)
    obs, _ = env.reset(seed=1, options={"segment": 1})
    assert env._episode_start == 240
    for _ in range(50):
        obs, reward, done, _, info = env.step(env.action_space.sample())
        if done:
            break
    print(f"env rollout ok, portfolio {info['portfolio_value']:.2f}")

    # --- baselines + metrics ---
    bh = summarize(buy_and_hold(val_df))
    ma = summarize(ma_crossover(val_df))
    print(f"baselines ok: B&H sharpe {bh['sharpe']:.3f}, MA sharpe {ma['sharpe']:.3f}")

    # --- engine, all three modes (+ the regret adversary), tiny budgets ---
    with tempfile.TemporaryDirectory() as tmp:
        for mode, env_fitness in [("single", "neg_sharpe"), ("population", "neg_sharpe"),
                                  ("coevolution", "neg_sharpe"), ("coevolution", "regret")]:
            config = EngineConfig(
                mode=mode, seed=7, generations=2,
                n_traders=3, n_env=4,
                matches_per_trader=2, segments_per_match=2,
                timesteps_per_match=512, hof_matches=1,
                env_fitness=env_fitness,
            )
            out_dir = os.path.join(tmp, f"{mode}_{env_fitness}")
            engine = CoevolutionEngine(config, pool, FEATURES, val_df, out_dir)
            _, model_path = engine.run(
                on_generation=lambda g, r, m=mode: print(
                    f"  [{m}] gen {g}: internal max "
                    f"{r['internal_trader_fitness_max']:+.3f}, "
                    f"external sharpe {r['external_val_sharpe']:+.3f}"))
            assert os.path.exists(model_path + ".zip")
            log = pd.read_csv(os.path.join(out_dir, "generations.csv"))
            assert len(log) == 2
            # every final trader is kept, not just the rank-0 pick
            idx = pd.read_csv(os.path.join(out_dir, "final_traders", "index.csv"))
            assert len(idx) == len(engine.traders)
            for p in idx["path"]:
                assert os.path.exists(os.path.join(out_dir, "final_traders", p))
            cfg_json = json.load(open(os.path.join(out_dir, "config.json")))
            assert "git_revision" in cfg_json
            print(f"mode '{mode}' ({env_fitness}) ok")

        # coevolution artifacts: HOFs should be non-empty
        assert len(engine.env_hof) > 0 and len(engine.trader_hof) > 0
        print("halls of fame populated")

        # --- the trader hall of fame must be a real snapshot -------------
        # 1. archived tensors must not alias any live trader's tensors
        live_ptrs = {
            t.data_ptr() for tr in engine.traders
            for t in tr.model.get_parameters()["policy"].values()
        }
        for entry in engine.trader_hof.entries:
            for t in entry.payload["policy"].values():
                assert t.data_ptr() not in live_ptrs, "HOF entry aliases a live trader"
        # 2. scoring adversaries against the archive must not touch trader 0
        engine._run_matches(gen=99)
        after = engine.traders[0].model.get_parameters()["policy"]
        # trader 0 trained during _run_matches, so weights legitimately moved;
        # what must NOT happen is trader 0 becoming a copy of an archived entry
        for entry in engine.trader_hof.entries:
            assert not all(torch.equal(entry.payload["policy"][k], after[k]) for k in after), \
                "trader 0 was overwritten by a hall-of-fame entry"
        # 3. the shell used for archive evaluation is not one of the traders
        assert engine._hof_shell.model is not engine.traders[0].model
        print("trader hall of fame is a genuine snapshot")

        # --- the overwrite guard --------------------------------------------
        try:
            CoevolutionEngine(config, pool, FEATURES, val_df, out_dir)
        except FileExistsError:
            print("overwrite guard ok")
        else:
            raise AssertionError("engine overwrote a populated out_dir without --overwrite")
        CoevolutionEngine(EngineConfig(**{**config.__dict__, "overwrite": True}),
                          pool, FEATURES, val_df, out_dir)
        print("explicit overwrite ok")

        # --- regret reference table: cash floor at 0 ----------------------
        assert engine._ref_by_start and min(engine._ref_by_start.values()) >= 0.0
        print("regret reference table ok")

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
