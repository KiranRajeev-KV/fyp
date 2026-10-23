"""Settings for one final BTC run; see ppo/README.md for the four-run matrix.

The evaluator fixes costs, folds, segments, and the pass rule. This node chooses
features, seeds, engine overrides, and whether populations vote or one saved
best trader is scored.
"""

CONFIG = {
    "name": "r21a-pure-alp", "score": "vote",
    "seeds": [42, 43, 44, 45, 46, 47],
    "flow_pos": True,          # used for observation inputs only when explicit features are absent
    "features": ["volatility_168h", "ema50_ema200_ratio"],   # observation inputs; forecasts still use base/flow predictors
    "rhat": [4, 24, 72],       # ridge forecast horizons (int = one feature "rhat", 0 = off)
    "engine": {                # EngineConfig overrides, all arms
        "generations": 12, "n_traders": 6, "n_env": 12, "matches_per_trader": 2,
        "segments_per_match": 4, "timesteps_per_match": 4096, "hof_matches": 1,
        "env_hof_matches": 1, "gamma": 0.997, "ent_coef_max": 0.02,
        "env_fresh_frac": 0.25, "env_weight_sigma": 0.7, "env_temp_sigma": 0.35,
        "trade_penalty": 0.0, "decision_every": 4, "start_long": True, "target_position": True,
    },
    "arm_overrides": {"coevolution": {"env_fitness": "progress", "adv_regime_lock": True, "shared_exam": 8, "env_adversary_frac": 0.5}},
    # runs r18a / r19a also add an input-drift adversary to the coevolution arm:
    #   "obs_adversary": ["rhat4", "rhat24", "rhat72"], "obs_bound": [0.75, 0.4], "obs_sign_random": True
}
