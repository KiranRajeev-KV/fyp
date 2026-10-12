"""The ONE file an orx node varies (COEVO V7 GOAL). coevo_eval.py is fixed."""

CONFIG = {
    "name": "smoke",
    "seeds": [42],
    "flow_pos": True,          # add order-flow + positioning features
    "rhat": 4,                 # ridge forecast feature horizon (0 = off)
    "engine": {                # EngineConfig overrides, all arms
        "generations": 1, "n_traders": 2, "n_env": 4, "matches_per_trader": 1,
        "segments_per_match": 2, "timesteps_per_match": 1024, "hof_matches": 1,
        "env_hof_matches": 1, "gamma": 0.997, "ent_coef_max": 0.02,
        "trade_penalty": 0.0, "decision_every": 4, "start_long": True,
    },
    "arm_overrides": {},       # e.g. {"coevolution": {"shared_exam": 8}}
}
