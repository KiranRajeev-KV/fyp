"""The ONE file an orx node varies (COEVO V7 GOAL). coevo_eval.py is fixed."""

CONFIG = {
    "name": "r2a-multi-horizon-forecasts",
    "seeds": [42, 43, 44],
    "flow_pos": True,          # add order-flow + positioning features
    "rhat": [4, 24, 72],       # ridge forecast horizons (int = one feature "rhat", 0 = off)
    "engine": {                # EngineConfig overrides, all arms
        "generations": 12, "n_traders": 6, "n_env": 12, "matches_per_trader": 2,
        "segments_per_match": 4, "timesteps_per_match": 4096, "hof_matches": 1,
        "env_hof_matches": 1, "gamma": 0.997, "ent_coef_max": 0.02,
        "env_fresh_frac": 0.25, "env_weight_sigma": 0.7, "env_temp_sigma": 0.35,
        "trade_penalty": 0.0, "decision_every": 4, "start_long": True,
    },
    "arm_overrides": {"coevolution": {"env_fitness": "regret"}},       # e.g. {"coevolution": {"shared_exam": 8}}
}
