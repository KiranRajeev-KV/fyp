"""End-to-end smoke test on synthetic data.

Generates a small synthetic OHLCV series (GBM with regime shifts), builds
the segment pool, runs a tiny run of each engine mode for 2 generations,
and runs the evaluation harness pieces. Verifies the machinery executes —
it says nothing about trading performance.

Run (from ppo/tests, with the ppo/.venv active):
    python smoke_test.py
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd

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

    # --- engine, all three modes, tiny budgets ---
    with tempfile.TemporaryDirectory() as tmp:
        for mode in ["single", "population", "coevolution"]:
            config = EngineConfig(
                mode=mode, seed=7, generations=2,
                n_traders=3, n_env=4,
                matches_per_trader=2, segments_per_match=2,
                timesteps_per_match=512, hof_matches=1,
            )
            engine = CoevolutionEngine(
                config, pool, FEATURES, val_df, os.path.join(tmp, mode))
            _, model_path = engine.run(
                on_generation=lambda g, r, m=mode: print(
                    f"  [{m}] gen {g}: internal max "
                    f"{r['internal_trader_fitness_max']:+.3f}, "
                    f"external sharpe {r['external_val_sharpe']:+.3f}"))
            assert os.path.exists(model_path + ".zip")
            log = pd.read_csv(os.path.join(tmp, mode, "generations.csv"))
            assert len(log) == 2
            print(f"mode '{mode}' ok")

        # coevolution artifacts: HOFs should be non-empty
        assert len(engine.env_hof) > 0 and len(engine.trader_hof) > 0
        print("halls of fame populated")

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
