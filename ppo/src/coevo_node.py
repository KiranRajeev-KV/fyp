"""The ONE file an orx node varies (COEVO V7 GOAL). coevo_eval.py is fixed."""

CONFIG = {
    "name": "r18a-progress-6-seeds", "score": "vote",
    "seeds": [42, 43, 44, 45, 46, 47],
    "flow_pos": True,          # add order-flow + positioning features
    "features": ["volatility_168h", "ema50_ema200_ratio"],   # explicit inputs (overrides the above)
    "rhat": [4, 24, 72],       # ridge forecast horizons (int = one feature "rhat", 0 = off)
    "engine": {                # EngineConfig overrides, all arms
        "generations": 12, "n_traders": 6, "n_env": 12, "matches_per_trader": 2,
        "segments_per_match": 4, "timesteps_per_match": 4096, "hof_matches": 1,
        "env_hof_matches": 1, "gamma": 0.997, "ent_coef_max": 0.02,
        "env_fresh_frac": 0.25, "env_weight_sigma": 0.7, "env_temp_sigma": 0.35,
        "trade_penalty": 0.0, "decision_every": 4, "start_long": True, "target_position": True,
    },
    "arm_overrides": {"coevolution": {"env_fitness": "progress", "adv_regime_lock": True, "shared_exam": 8, "env_adversary_frac": 0.5,
                                      "obs_bound": (0.75, 0.4), "obs_sign_random": True, "obs_adversary": ["rhat4", "rhat24", "rhat72"]}},
}
