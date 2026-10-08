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


def make_bounds(ent_coef_max=None, ent_coef_min=None):
    """Copy of HYPERPARAM_BOUNDS with the ent_coef range optionally narrowed.

    High ent_coef (up to 0.1) drives random action-flipping, which the diagnostic
    tied to churn-driven losses, so capping it (e.g. 0.02) lets PBT prefer calmer
    policies. The opposite failure is the coevolution arm's: entropy decays, the
    policy settles on "hold" and never re-enters the market, so a floor
    (e.g. 0.005) keeps some exploration alive. 0/None leaves that end alone.
    """
    lo, hi = HYPERPARAM_BOUNDS["ent_coef"]
    return {**HYPERPARAM_BOUNDS,
            "ent_coef": (ent_coef_min or lo, ent_coef_max or hi)}


def sample_hyperparams(rng, bounds=HYPERPARAM_BOUNDS):
    return {
        name: float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        for name, (lo, hi) in bounds.items()
    }


def crossover_hyperparams(a, b, rng, bounds=HYPERPARAM_BOUNDS):
    return {name: (a if rng.uniform() < 0.5 else b)[name] for name in bounds}


def mutate_hyperparams(hp, rng, factor=1.25, bounds=HYPERPARAM_BOUNDS):
    out = {}
    for name, (lo, hi) in bounds.items():
        scale = factor if rng.uniform() < 0.5 else 1.0 / factor
        out[name] = float(np.clip(hp[name] * scale, lo, hi))
    return out


@dataclass
class Trader:
    trader_id: int
    hyperparams: dict
    model: PPO = None
    fitness: float = 0.0
    # provenance: how this individual's *current* hyperparams came about.
    # "init" at creation, then set by pbt_step each generation to either
    # "survived" or "exploit<-{parent_id}". Logged so the hyperparameter
    # trajectory can be read as a causal chain (fitness -> rank -> change).
    origin: str = "init"
    no_trade_frac: float = 0.0   # share of evaluated segments with zero trades
    _scores: list = field(default_factory=list)
    _no_trade: list = field(default_factory=list)

    def apply_hyperparams(self):
        self.model.learning_rate = self.hyperparams["learning_rate"]
        self.model._setup_lr_schedule()
        self.model.ent_coef = self.hyperparams["ent_coef"]
        self.model.clip_range = _clip_schedule(self.hyperparams["clip_range"])

    def record(self, score, no_trade_frac=None):
        self._scores.append(score)
        if no_trade_frac is not None:
            self._no_trade.append(no_trade_frac)

    def finalize_fitness(self):
        self.fitness = float(np.mean(self._scores)) if self._scores else 0.0
        self.no_trade_frac = float(np.mean(self._no_trade)) if self._no_trade else 0.0
        self._scores = []
        self._no_trade = []


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


def create_trader(trader_id, df, feature_columns, env_kwargs, seed, rng,
                  policy_kwargs=None, gamma=0.99, bounds=HYPERPARAM_BOUNDS):
    hp = sample_hyperparams(rng, bounds)
    env = make_env(df, feature_columns, None, env_kwargs, seed=seed)
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=hp["learning_rate"],
        ent_coef=hp["ent_coef"],
        clip_range=hp["clip_range"],
        n_steps=1024,
        batch_size=256,
        gamma=gamma,          # higher gamma => longer horizon (value multi-week trends)
        gae_lambda=0.95,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=seed,
        verbose=0,
        policy_kwargs=policy_kwargs,
    )
    return Trader(trader_id=trader_id, hyperparams=hp, model=model)


PPO_DIAGNOSTICS = [
    "train/clip_fraction",          # 0 => clip_range never binds, i.e. is inert
    "train/approx_kl",              # how far each update actually moved the policy
    "train/explained_variance",     # <=0 => the value head is learning nothing
    "train/entropy_loss",           # how random the policy still is
    "train/value_loss",
    "train/policy_gradient_loss",
]


def ppo_diagnostics(trader):
    """Latest SB3 training diagnostics for this trader, as a flat dict."""
    vals = getattr(trader.model, "logger", None)
    vals = getattr(vals, "name_to_value", {}) if vals is not None else {}
    return {k.split("/", 1)[1]: float(vals[k]) for k in PPO_DIAGNOSTICS if k in vals}


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
        trades = 0
        in_market = 0
        steps = 0
        while not done:
            action, _ = trader.model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(action)
            values.append(info["portfolio_value"])
            trades = info["trade_count"]
            in_market += 1 if info["btc_held"] > 0 else 0
            steps += 1
        metrics = summarize(values)
        metrics["exposure"] = float(in_market / steps) if steps else 0.0
        # A trader that never trades holds a flat portfolio, so the return std
        # is 0 and sharpe_ratio returns its documented 0.0. That is a fine
        # convention, but it makes "did nothing" indistinguishable from "traded
        # and broke even" -- and the former says nothing about the adversary
        # that served the window. Record the distinction rather than changing
        # the metric, which every published number depends on.
        metrics["trades"] = trades
        metrics["no_trade"] = trades == 0
        per_segment.append(metrics)
    score = float(np.mean([m[fitness_metric] for m in per_segment]))
    return score, per_segment


def pbt_step(traders, rng, exploit_frac=0.25, bounds=HYPERPARAM_BOUNDS):
    """Bottom quantile copies weights from a top-quantile parent, takes
    hyperparams from a crossover of two top parents, then mutates them.

    Returns the list of exploit events (dicts) so the caller can log which
    hyperparameter values were replaced by which, and why.
    """
    ranked = sorted(traders, key=lambda t: t.fitness, reverse=True)
    n = len(ranked)
    n_exploit = max(1, int(n * exploit_frac))
    if n < 2:
        return []

    for t in ranked:
        t.origin = "survived"

    tops = ranked[:max(2, n_exploit)]
    events = []
    for loser in ranked[-n_exploit:]:
        parent_a, parent_b = rng.choice(len(tops), size=2, replace=False)
        parent_a, parent_b = tops[parent_a], tops[parent_b]

        before = dict(loser.hyperparams)
        loser.model.set_parameters(parent_a.model.get_parameters())
        crossed = crossover_hyperparams(
            parent_a.hyperparams, parent_b.hyperparams, rng, bounds)
        loser.hyperparams = mutate_hyperparams(crossed, rng, bounds=bounds)
        loser.apply_hyperparams()
        loser.origin = f"exploit<-{parent_a.trader_id}"

        events.append({
            "trader_id": loser.trader_id,
            "weights_from": parent_a.trader_id,
            "hp_parent_a": parent_a.trader_id,
            "hp_parent_b": parent_b.trader_id,
            "before": before,
            "crossed": crossed,
            "after": dict(loser.hyperparams),
        })
    return events
