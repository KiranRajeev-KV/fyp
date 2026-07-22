"""Competitive coevolutionary engine.

One engine, three modes — which IS the paper's three-way ablation:
  single      — 1 trader, uniform random segments, no evolution (control)
  population  — trader population + PBT, uniform random segments (no adversary)
  coevolution — trader population + PBT vs. evolving environment population

Every generation logs two ledgers separately (the anti-Red-Queen guardrail):
  internal — relative fitness inside the arms race (proves nothing alone)
  external — best trader evaluated on fixed held-out segments against fixed
             baselines (the objective yardstick).
"""

import csv
import json
import os
from dataclasses import dataclass, asdict
from types import SimpleNamespace

import numpy as np

from baselines import buy_and_hold, ma_crossover
from metrics import summarize
from coevo import env_population as ep
from coevo import trader_population as tp
from coevo.hall_of_fame import HallOfFame


@dataclass
class EngineConfig:
    mode: str = "coevolution"          # single | population | coevolution
    seed: int = 42
    generations: int = 20
    n_traders: int = 8
    n_env: int = 12
    matches_per_trader: int = 3        # env opponents faced per generation
    segments_per_match: int = 4
    timesteps_per_match: int = 8192
    hof_matches: int = 1               # trader extra matches vs archived adversaries
    env_hof_matches: int = 1           # env extra matches vs archived traders
    hof_capacity: int = 20
    fitness_metric: str = "sharpe"
    exploit_frac: float = 0.25
    elite_frac: float = 0.25
    sharing_strength: float = 1.0
    initial_balance: float = 10000.0
    trading_fee: float = 0.001
    slippage: float = 0.0005


class CoevolutionEngine:

    def __init__(self, config, pool, feature_columns, val_df, out_dir):
        self.cfg = config
        self.pool = pool
        self.feature_columns = feature_columns
        self.val_df = val_df.reset_index(drop=True)
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.rng = np.random.default_rng(config.seed)
        self.env_kwargs = {
            "initial_balance": config.initial_balance,
            "trading_fee": config.trading_fee,
            "slippage": config.slippage,
        }

        n_traders = 1 if config.mode == "single" else config.n_traders
        self.traders = [
            tp.create_trader(
                trader_id=i,
                df=pool.df,
                feature_columns=feature_columns,
                env_kwargs=self.env_kwargs,
                seed=config.seed * 1000 + i,
                rng=self.rng,
            )
            for i in range(n_traders)
        ]

        self.env_pop = (
            [ep.random_individual(self.rng) for _ in range(config.n_env)]
            if config.mode == "coevolution" else None
        )

        self.env_hof = HallOfFame(config.hof_capacity, self.rng)
        self.trader_hof = HallOfFame(config.hof_capacity, self.rng)

        # external yardstick: fixed val segments + fixed baselines, computed once
        self.val_segments = [(0, len(self.val_df))]
        self.baseline_metrics = {
            "buy_hold": summarize(buy_and_hold(
                self.val_df, config.initial_balance, config.trading_fee)),
            "ma_crossover": summarize(ma_crossover(
                self.val_df, config.initial_balance, config.trading_fee)),
        }

        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(asdict(config), f, indent=2)

        self.log_path = os.path.join(out_dir, "generations.csv")
        self._log_fields = [
            "generation",
            "internal_trader_fitness_mean", "internal_trader_fitness_max",
            "internal_env_fitness_mean", "internal_env_fitness_max",
            "external_val_sharpe", "external_val_return_pct",
            "external_val_max_drawdown_pct",
            "baseline_bh_sharpe", "baseline_ma_sharpe",
        ]
        with open(self.log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self._log_fields).writeheader()

    # ------------------------------------------------------------------
    # segment selection per mode
    # ------------------------------------------------------------------

    def _uniform_segments(self, n):
        idx = self.rng.choice(len(self.pool), size=min(n, len(self.pool)), replace=False)
        return [(self.pool.segments[i].start, self.pool.segments[i].end) for i in idx]

    # ------------------------------------------------------------------
    # one generation
    # ------------------------------------------------------------------

    def _run_matches(self):
        cfg = self.cfg
        for trader in self.traders:
            if self.env_pop is not None:
                k = min(cfg.matches_per_trader, len(self.env_pop))
                opponent_idx = self.rng.choice(len(self.env_pop), size=k, replace=False)
                opponents = [self.env_pop[i] for i in opponent_idx]
            else:
                opponents = [None] * cfg.matches_per_trader

            for opp in opponents:
                segs = (
                    opp.select_segments(self.pool, cfg.segments_per_match, self.rng)
                    if opp is not None
                    else self._uniform_segments(cfg.segments_per_match)
                )
                tp.train_on_segments(
                    trader, self.pool.df, self.feature_columns, segs,
                    self.env_kwargs, cfg.timesteps_per_match,
                )
                score, _ = tp.evaluate_on_segments(
                    trader, self.pool.df, self.feature_columns, segs,
                    self.env_kwargs, cfg.fitness_metric,
                )
                trader.record(score)
                if opp is not None:
                    opp.record(score)

            # matches against archived adversaries: evaluation only, so the
            # trader cannot forget how to beat old opponents (anti-cycling)
            for entry in self.env_hof.sample(cfg.hof_matches):
                segs = entry.payload.select_segments(self.pool, cfg.segments_per_match, self.rng)
                score, _ = tp.evaluate_on_segments(
                    trader, self.pool.df, self.feature_columns, segs,
                    self.env_kwargs, cfg.fitness_metric,
                )
                trader.record(score)

        # symmetric anti-cycling: adversaries are also scored against
        # archived trader snapshots, so an env individual can't look strong
        # merely by exploiting a quirk of the current generation's traders
        if self.env_pop is not None and len(self.trader_hof) > 0:
            shell = SimpleNamespace(model=self.traders[0].model)
            for ind in self.env_pop:
                for entry in self.trader_hof.sample(cfg.env_hof_matches):
                    live_params = shell.model.get_parameters()
                    shell.model.set_parameters(entry.payload)
                    segs = ind.select_segments(self.pool, cfg.segments_per_match, self.rng)
                    score, _ = tp.evaluate_on_segments(
                        shell, self.pool.df, self.feature_columns, segs,
                        self.env_kwargs, cfg.fitness_metric,
                    )
                    shell.model.set_parameters(live_params)
                    ind.record(score)

    def _external_eval(self, best_trader):
        _, per_segment = tp.evaluate_on_segments(
            best_trader, self.val_df, self.feature_columns, self.val_segments,
            self.env_kwargs, self.cfg.fitness_metric,
        )
        return per_segment[0]

    def run(self, on_generation=None):
        cfg = self.cfg
        for gen in range(cfg.generations):
            self._run_matches()

            for trader in self.traders:
                trader.finalize_fitness()

            if self.env_pop is not None:
                for ind in self.env_pop:
                    ind.finalize_raw_fitness()
                ep.apply_fitness_sharing(
                    self.env_pop, self.pool.descriptor_matrix, cfg.sharing_strength)

            best_trader = max(self.traders, key=lambda t: t.fitness)
            external = self._external_eval(best_trader)

            self.trader_hof.add(
                best_trader.model.get_parameters(), best_trader.fitness, gen)
            if self.env_pop is not None:
                best_env = max(self.env_pop, key=lambda e: e.fitness)
                self.env_hof.add(
                    ep.EnvIndividual(
                        weights=best_env.weights.copy(),
                        temperature=best_env.temperature),
                    best_env.fitness, gen)

            row = {
                "generation": gen,
                "internal_trader_fitness_mean":
                    float(np.mean([t.fitness for t in self.traders])),
                "internal_trader_fitness_max": best_trader.fitness,
                "internal_env_fitness_mean":
                    float(np.mean([e.fitness for e in self.env_pop]))
                    if self.env_pop else float("nan"),
                "internal_env_fitness_max":
                    max(e.fitness for e in self.env_pop)
                    if self.env_pop else float("nan"),
                "external_val_sharpe": external["sharpe"],
                "external_val_return_pct": external["total_return_pct"],
                "external_val_max_drawdown_pct": external["max_drawdown_pct"],
                "baseline_bh_sharpe": self.baseline_metrics["buy_hold"]["sharpe"],
                "baseline_ma_sharpe": self.baseline_metrics["ma_crossover"]["sharpe"],
            }
            with open(self.log_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=self._log_fields).writerow(row)
            if on_generation:
                on_generation(gen, row)

            # variation — after logging so the logged fitness matches this gen
            if self.env_pop is not None:
                self.env_pop = ep.evolve(
                    self.env_pop, self.rng,
                    elite_frac=cfg.elite_frac)
            if cfg.mode != "single":
                tp.pbt_step(self.traders, self.rng, cfg.exploit_frac)

        best_trader = max(self.traders, key=lambda t: t.fitness)
        model_path = os.path.join(self.out_dir, "best_trader")
        best_trader.model.save(model_path)
        return best_trader, model_path
