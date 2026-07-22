"""Trader population: PPO agents evolved via Population-Based Training.

Crossover/mutation act on *hyperparameters only* (learning rate, entropy
coefficient, clip range) — never on network weights, which are destroyed by
weight-space crossover (permutation symmetry). PPO's gradient does all the
weight learning inside each individual; PBT exploit copies weights wholesale
from a stronger individual.
"""

from dataclasses import dataclass, field

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
try:
    from stable_baselines3.common.utils import FloatSchedule as _clip_schedule
except ImportError:  # older SB3
    from stable_baselines3.common.utils import get_schedule_fn as _clip_schedule

from trading_env import TradingEnv
from metrics import summarize

HYPERPARAM_BOUNDS = {
    "learning_rate": (1e-5, 1e-3),
    "ent_coef": (1e-4, 0.1),
    "clip_range": (0.05, 0.4),
}


def sample_hyperparams(rng):
    return {
        name: float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        for name, (lo, hi) in HYPERPARAM_BOUNDS.items()
    }


def crossover_hyperparams(a, b, rng):
    return {name: (a if rng.uniform() < 0.5 else b)[name] for name in HYPERPARAM_BOUNDS}


def mutate_hyperparams(hp, rng, factor=1.25):
    out = {}
    for name, (lo, hi) in HYPERPARAM_BOUNDS.items():
        scale = factor if rng.uniform() < 0.5 else 1.0 / factor
        out[name] = float(np.clip(hp[name] * scale, lo, hi))
    return out


@dataclass
class Trader:
    trader_id: int
    hyperparams: dict
    model: PPO = None
    fitness: float = 0.0
    _scores: list = field(default_factory=list)

    def apply_hyperparams(self):
        self.model.learning_rate = self.hyperparams["learning_rate"]
        self.model._setup_lr_schedule()
        self.model.ent_coef = self.hyperparams["ent_coef"]
        self.model.clip_range = _clip_schedule(self.hyperparams["clip_range"])

    def record(self, score):
        self._scores.append(score)

    def finalize_fitness(self):
        self.fitness = float(np.mean(self._scores)) if self._scores else 0.0
        self._scores = []


def make_env(df, feature_columns, segments, env_kwargs, seed=None):
    env = TradingEnv(
        df=df,
        feature_columns=feature_columns,
        segments=segments,
        **env_kwargs,
    )
    if seed is not None:
        env.reset(seed=seed)
    return Monitor(env)


def create_trader(trader_id, df, feature_columns, env_kwargs, seed, rng, policy_kwargs=None):
    hp = sample_hyperparams(rng)
    env = make_env(df, feature_columns, None, env_kwargs, seed=seed)
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=hp["learning_rate"],
        ent_coef=hp["ent_coef"],
        clip_range=hp["clip_range"],
        n_steps=1024,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=seed,
        verbose=0,
        policy_kwargs=policy_kwargs,
    )
    return Trader(trader_id=trader_id, hyperparams=hp, model=model)


def train_on_segments(trader, df, feature_columns, segments, env_kwargs, timesteps, seed=None):
    env = make_env(df, feature_columns, segments, env_kwargs, seed=seed)
    trader.model.set_env(env)
    trader.model.learn(total_timesteps=timesteps, reset_num_timesteps=False, progress_bar=False)


def evaluate_on_segments(trader, df, feature_columns, segments, env_kwargs,
                         fitness_metric="sharpe"):
    """Deterministic rollout on each segment; returns (mean fitness, per-segment metrics)."""
    per_segment = []
    for i, _ in enumerate(segments):
        env = TradingEnv(
            df=df,
            feature_columns=feature_columns,
            segments=segments,
            **env_kwargs,
        )
        obs, _ = env.reset(seed=0, options={"segment": i})
        values = [env_kwargs.get("initial_balance", 10000)]
        done = False
        while not done:
            action, _ = trader.model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(action)
            values.append(info["portfolio_value"])
        per_segment.append(summarize(values))
    score = float(np.mean([m[fitness_metric] for m in per_segment]))
    return score, per_segment


def pbt_step(traders, rng, exploit_frac=0.25):
    """Bottom quantile copies weights from a top-quantile parent, takes
    hyperparams from a crossover of two top parents, then mutates them."""
    ranked = sorted(traders, key=lambda t: t.fitness, reverse=True)
    n = len(ranked)
    n_exploit = max(1, int(n * exploit_frac))
    if n < 2:
        return

    tops = ranked[:max(2, n_exploit)]
    for loser in ranked[-n_exploit:]:
        parent_a, parent_b = rng.choice(len(tops), size=2, replace=False)
        parent_a, parent_b = tops[parent_a], tops[parent_b]

        loser.model.set_parameters(parent_a.model.get_parameters())
        loser.hyperparams = mutate_hyperparams(
            crossover_hyperparams(parent_a.hyperparams, parent_b.hyperparams, rng),
            rng,
        )
        loser.apply_hyperparams()
