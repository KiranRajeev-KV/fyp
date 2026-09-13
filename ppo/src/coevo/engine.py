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
import subprocess
import time
from dataclasses import dataclass, asdict

import numpy as np

from baselines import buy_and_hold, ma_crossover
from metrics import summarize
from coevo import env_population as ep
from coevo import trader_population as tp
from coevo.hall_of_fame import HallOfFame, snapshot_parameters
from segments import DESCRIPTOR_NAMES


def git_revision(path):
    """Commit hash (+ '-dirty' if the tree has uncommitted changes) or None."""
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, stderr=subprocess.DEVNULL,
            text=True).strip()
        dirty = subprocess.run(
            ["git", "diff", "--quiet", "HEAD"], cwd=path,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0
        return rev + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


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
    env_weight_sigma: float = 0.3      # adversary weight mutation scale
    env_temp_sigma: float = 0.15       # adversary temperature mutation scale
    env_fresh_frac: float = 0.0        # fraction of fresh random adversaries per gen
    tournament_k: int = 3              # adversary tournament selection size
    initial_balance: float = 10000.0
    trading_fee: float = 0.001
    slippage: float = 0.0005
    # anti-churn / horizon fixes (defaults reproduce the original behavior)
    trade_penalty: float = 0.0         # extra reward penalty per executed trade
    gamma: float = 0.99                # PPO discount; raise (0.995+) for longer horizon
    ent_coef_max: float = 0.1          # upper bound on evolved entropy coef
    # A match in which the trader never traded on ANY segment scores Sharpe 0
    # for every adversary that served it, so it measures the trader's collapse,
    # not the adversary's skill -- and it is common-mode noise across the whole
    # adversary population, which corrupts their relative ranking. Enable to
    # drop such matches from adversary fitness. Default False reproduces the
    # original behaviour (and the published V1-V3 numbers).
    env_skip_no_trade_matches: bool = False
    # How the adversary is scored on each segment it served:
    #   neg_sharpe -- the original: -(trader fitness metric on that segment).
    #                 Bounded above by 0 once the trader learns to hold cash on
    #                 hard windows (a flat portfolio scores exactly 0), so the
    #                 adversary's landscape has no gradient left late in a run.
    #   regret     -- max(0, buy&hold, MA-crossover) on that segment minus the
    #                 trader's score: "where does the trader do worse than a
    #                 trivial policy?". Unbounded, and it moves as the trader
    #                 learns instead of terminating.
    env_fitness: str = "neg_sharpe"
    # Refuse to start in an out_dir that already holds a populated
    # generations.csv (a relaunch destroyed the V4 ledgers). True = overwrite.
    overwrite: bool = False


class CoevolutionEngine:

    def __init__(self, config, pool, feature_columns, val_df, out_dir):
        self.cfg = config
        self.pool = pool
        self.feature_columns = feature_columns
        self.val_df = val_df.reset_index(drop=True)
        self.out_dir = out_dir
        existing = os.path.join(out_dir, "generations.csv")
        if os.path.exists(existing) and not config.overwrite:
            with open(existing) as f:
                n_rows = sum(1 for _ in f) - 1
            if n_rows > 0:
                raise FileExistsError(
                    f"{out_dir} already holds a run with {n_rows} logged "
                    f"generation(s). Pass overwrite=True / --overwrite to "
                    f"replace it, or choose another --out-root.")
        os.makedirs(out_dir, exist_ok=True)

        self.rng = np.random.default_rng(config.seed)
        self.env_kwargs = {
            "initial_balance": config.initial_balance,
            "trading_fee": config.trading_fee,
            "slippage": config.slippage,
            "trade_penalty": config.trade_penalty,
        }
        # entropy-capped hyperparameter bounds (anti-churn); None ent_coef_max
        # keeps the original 0.1 ceiling
        self.bounds = tp.make_bounds(
            config.ent_coef_max if config.ent_coef_max < 0.1 else None)

        n_traders = 1 if config.mode == "single" else config.n_traders
        self.traders = [
            tp.create_trader(
                trader_id=i,
                df=pool.df,
                feature_columns=feature_columns,
                env_kwargs=self.env_kwargs,
                seed=config.seed * 1000 + i,
                rng=self.rng,
                gamma=config.gamma,
                bounds=self.bounds,
            )
            for i in range(n_traders)
        ]

        self.env_pop = (
            [ep.random_individual(self.rng) for _ in range(config.n_env)]
            if config.mode == "coevolution" else None
        )

        self.env_hof = HallOfFame(config.hof_capacity, self.rng)
        self.trader_hof = HallOfFame(config.hof_capacity, self.rng)
        # A throwaway network that archived trader snapshots are loaded into
        # when adversaries are scored against the hall of fame. Previously
        # trader 0 was borrowed for this, and because SB3 parameter dicts alias
        # the live tensors the "restore" was a no-op: trader 0 left every
        # generation as a clone of the last archived entry. Never train this.
        self._hof_shell = (
            tp.create_trader(
                trader_id=-1, df=pool.df, feature_columns=feature_columns,
                env_kwargs=self.env_kwargs, seed=config.seed * 1000 + 999,
                rng=np.random.default_rng(config.seed), gamma=config.gamma,
                bounds=self.bounds)
            if config.mode == "coevolution" else None
        )

        if config.env_fitness not in ("neg_sharpe", "regret"):
            raise ValueError(f"unknown env_fitness {config.env_fitness!r}")
        # regret reference per pool segment: the best trivial policy's score
        # (cash = 0, buy & hold, MA-crossover) under the same fee model
        self._ref_by_start = {}
        if config.env_fitness == "regret" and self.env_pop is not None:
            for sg in pool.segments:
                seg_df = pool.slice(sg)
                bh = summarize(buy_and_hold(
                    seg_df, config.initial_balance, config.trading_fee, config.slippage))
                ma = summarize(ma_crossover(
                    seg_df, config.initial_balance, config.trading_fee, slippage=config.slippage))
                self._ref_by_start[sg.start] = max(
                    0.0, bh[config.fitness_metric], ma[config.fitness_metric])

        # which regime each segment belongs to, so the curriculum the adversary
        # actually served can be logged (not just the genome that chose it)
        self._regime_by_start = {sg.start: sg.regime for sg in pool.segments}
        # a fixed probe batch for measuring how far apart the policies are.
        # Seeded independently of the run so it is identical across seeds.
        self._probe = self._build_probe()

        # external yardstick: fixed val segments + fixed baselines, computed once
        self.val_segments = [(0, len(self.val_df))]
        self.baseline_metrics = {
            "buy_hold": summarize(buy_and_hold(
                self.val_df, config.initial_balance, config.trading_fee,
                config.slippage)),
            "ma_crossover": summarize(ma_crossover(
                self.val_df, config.initial_balance, config.trading_fee,
                slippage=config.slippage)),
        }

        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump({**asdict(config),
                       "git_revision": git_revision(os.path.dirname(__file__))},
                      f, indent=2)

        self.log_path = os.path.join(out_dir, "generations.csv")
        self._log_fields = [
            "generation",
            "internal_trader_fitness_mean", "internal_trader_fitness_max",
            "internal_env_fitness_mean", "internal_env_fitness_max",
            "external_val_sharpe", "external_val_return_pct",
            "external_val_max_drawdown_pct",
            "baseline_bh_sharpe", "baseline_ma_sharpe",
            # appended in this order; existing columns above are unchanged
            "external_val_sharpe_mean", "external_val_sharpe_std",
            "trader_exposure_mean", "trader_no_trade_frac_mean",
            "policy_disagreement", "identical_policy_pairs",
            "clip_fraction_mean", "explained_variance_mean",
            "seconds",
        ]
        with open(self.log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self._log_fields).writeheader()

        # ------------------------------------------------------------------
        # parameter ledgers: what the *search* is doing, generation by
        # generation. generations.csv records outcomes (fitness, external
        # Sharpe); these record the hyperparameters and adversary genomes that
        # produced them, so the two can be correlated after the fact.
        # ------------------------------------------------------------------
        self.trader_log_path = os.path.join(out_dir, "trader_params.csv")
        self._trader_log_fields = [
            "generation", "trader_id", "rank", "fitness",
            "learning_rate", "ent_coef", "clip_range", "no_trade_frac", "origin",
            "fitness_std", "n_matches", "exposure", "trades",
        ] + [k.split("/", 1)[1] for k in tp.PPO_DIAGNOSTICS]
        with open(self.trader_log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self._trader_log_fields).writeheader()

        self.pbt_log_path = os.path.join(out_dir, "pbt_events.csv")
        self._pbt_log_fields = (
            ["generation", "trader_id", "weights_from",
             "hp_parent_a", "hp_parent_b"]
            + [f"{name}_{stage}" for name in self.bounds
               for stage in ("before", "crossed", "after")]
        )
        with open(self.pbt_log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self._pbt_log_fields).writeheader()

        # one row per (generation, trader, match): the spread of scores that a
        # single fitness number averages away, the behaviour behind each score,
        # and which windows were actually served
        self.match_log_path = os.path.join(out_dir, "matches.csv")
        self._match_log_fields = [
            "generation", "trader_id", "match_idx", "opponent_kind", "opponent_id",
            "score", "trades", "exposure", "return_pct", "max_drawdown_pct",
            "no_trade_segments", "n_segments", "counted_for_adversary",
            "segment_starts", "segment_regimes",
        ]
        with open(self.match_log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self._match_log_fields).writeheader()

        # held-out performance for EVERY trader, not just the best one, so the
        # internal-vs-external relationship can be measured within a generation
        # (n_traders points per generation) instead of one point per generation
        self.ext_log_path = os.path.join(out_dir, "trader_external.csv")
        self._ext_log_fields = [
            "generation", "trader_id", "rank", "fitness",
            "external_sharpe", "external_return_pct", "external_max_drawdown_pct",
            "external_trades", "external_exposure",
        ]
        with open(self.ext_log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self._ext_log_fields).writeheader()

        self.env_log_path = os.path.join(out_dir, "adversary_params.csv")
        self._env_log_fields = (
            ["generation", "env_id", "rank", "raw_fitness", "shared_fitness",
             "niche", "temperature", "selection_entropy", "evaluated",
             "n_matches", "weight_norm",
             "frac_bear", "frac_bull", "frac_sideways"]
            + [f"w_{name}" for name in DESCRIPTOR_NAMES]
            + ["origin"]
        )
        with open(self.env_log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self._env_log_fields).writeheader()

    # ------------------------------------------------------------------
    # parameter ledgers
    # ------------------------------------------------------------------

    def _build_probe(self, n=256):
        """A fixed batch of observations used to compare policies directly.

        Half all-cash, half fully-invested, so both halves of the position
        space are probed. Seeded independently of the run seed so the probe is
        identical across seeds and runs, making the numbers comparable.
        """
        rng = np.random.default_rng(12345)
        feats = self.pool.df[self.feature_columns].values.astype(np.float32)
        idx = rng.choice(len(feats), size=min(n, len(feats)), replace=False)
        f = feats[idx]
        pos = np.zeros((len(f), 3), dtype=np.float32)
        half = len(f) // 2
        pos[:half] = (0.0, 1.0, 0.0)      # all cash
        pos[half:] = (1.0, 0.0, 1.0)      # fully in BTC
        return np.concatenate([f, pos], axis=1).astype(np.float32)

    def _policy_diversity(self):
        """How much do the traders actually still disagree?

        PBT copies whole networks, so the population can collapse to near
        clones. Hyperparameter spread does not show that; comparing the actions
        themselves does.
        """
        if len(self.traders) < 2:
            return {"policy_disagreement": float("nan"), "identical_policy_pairs": 0}
        acts = []
        for t in self.traders:
            a, _ = t.model.predict(self._probe, deterministic=True)
            acts.append(np.asarray(a).reshape(-1))
        dis, identical = [], 0
        for i in range(len(acts)):
            for j in range(i + 1, len(acts)):
                d = float(np.mean(acts[i] != acts[j]))
                dis.append(d)
                identical += (d == 0.0)
        return {"policy_disagreement": float(np.mean(dis)),
                "identical_policy_pairs": int(identical)}

    def _append(self, path, fields, rows):
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            for row in rows:
                writer.writerow(row)

    def _match_row(self, gen, trader, idx, kind, opp_id, segs, score, per_seg, counted):
        starts = [st for st, _ in segs]
        return {
            "generation": gen, "trader_id": trader.trader_id, "match_idx": idx,
            "opponent_kind": kind,
            "opponent_id": "" if opp_id is None else int(opp_id),
            "score": score,
            "trades": float(np.mean([m["trades"] for m in per_seg])),
            "exposure": float(np.mean([m["exposure"] for m in per_seg])),
            "return_pct": float(np.mean([m["total_return_pct"] for m in per_seg])),
            "max_drawdown_pct": float(np.mean([m["max_drawdown_pct"] for m in per_seg])),
            "no_trade_segments": int(sum(m["no_trade"] for m in per_seg)),
            "n_segments": len(per_seg),
            "counted_for_adversary": counted,
            "segment_starts": ";".join(str(x) for x in starts),
            "segment_regimes": ";".join(self._regime_by_start.get(x, "?") for x in starts),
        }

    def _log_trader_params(self, gen):
        ranked = sorted(self.traders, key=lambda t: t.fitness, reverse=True)
        rank_of = {id(t): r for r, t in enumerate(ranked)}
        # the per-match spread that a single fitness number averages away --
        # if this is large relative to the gaps between traders, ranking them
        # is ranking noise
        by_trader = {}
        for r in self._match_rows:
            by_trader.setdefault(r["trader_id"], []).append(r)
        self._append(self.trader_log_path, self._trader_log_fields, [
            {
                "generation": gen,
                "trader_id": t.trader_id,
                "rank": rank_of[id(t)],
                "fitness": t.fitness,
                "learning_rate": t.hyperparams["learning_rate"],
                "ent_coef": t.hyperparams["ent_coef"],
                "clip_range": t.hyperparams["clip_range"],
                # share of evaluated segments where this trader never traded --
                # a flat portfolio scores Sharpe 0, which looks like break-even
                "no_trade_frac": t.no_trade_frac,
                # how these values arose: "init", "survived", or
                # "exploit<-N" (set by last generation's pbt_step)
                "origin": t.origin,
                **self._trader_extras(by_trader.get(t.trader_id, []), t),
            }
            for t in self.traders
        ])

    def _trader_extras(self, rows, trader):
        scores = [r["score"] for r in rows]
        out = {
            "fitness_std": float(np.std(scores)) if len(scores) > 1 else 0.0,
            "n_matches": len(rows),
            "exposure": float(np.mean([r["exposure"] for r in rows])) if rows else float("nan"),
            "trades": float(np.mean([r["trades"] for r in rows])) if rows else float("nan"),
        }
        diag = tp.ppo_diagnostics(trader)
        for k in tp.PPO_DIAGNOSTICS:
            name = k.split("/", 1)[1]
            out[name] = diag.get(name, float("nan"))
        return out

    def _log_env_params(self, gen):
        dm = self.pool.descriptor_matrix
        ranked = sorted(self.env_pop, key=lambda e: e.fitness, reverse=True)
        rank_of = {id(e): r for r, e in enumerate(ranked)}
        rows = []
        for env_id, ind in enumerate(self.env_pop):
            row = {
                "generation": gen,
                "env_id": env_id,
                "rank": rank_of[id(ind)],
                "raw_fitness": ind.raw_fitness,
                "shared_fitness": ind.fitness,
                "niche": ind.niche,
                "temperature": ind.temperature,
                "selection_entropy": ind.selection_entropy(dm),
                # False => never drawn as an opponent, so raw_fitness 0.0 is a
                # placeholder, not a measurement; excluded from elites/parents
                "evaluated": ind.evaluated,
                **self._env_extras(env_id, ind),
                "origin": ind.origin,
            }
            row.update({f"w_{n}": float(w)
                        for n, w in zip(DESCRIPTOR_NAMES, ind.weights)})
            rows.append(row)
        self._append(self.env_log_path, self._env_log_fields, rows)

    def _env_extras(self, env_id, ind):
        served = self._env_served.get(env_id, {"n": 0, "regimes": []})
        regs = served["regimes"]
        n = len(regs) or 1
        return {
            "n_matches": served["n"],
            "weight_norm": float(np.linalg.norm(ind.weights)),
            # the curriculum this individual actually served, which is the
            # thing the "adversarial curriculum" hypothesis is about
            "frac_bear": regs.count("bear") / n if regs else float("nan"),
            "frac_bull": regs.count("bull") / n if regs else float("nan"),
            "frac_sideways": regs.count("sideways") / n if regs else float("nan"),
        }

    def _log_pbt_events(self, gen, events):
        rows = []
        for ev in events:
            row = {
                "generation": gen,
                "trader_id": ev["trader_id"],
                "weights_from": ev["weights_from"],
                "hp_parent_a": ev["hp_parent_a"],
                "hp_parent_b": ev["hp_parent_b"],
            }
            for name in self.bounds:
                for stage in ("before", "crossed", "after"):
                    row[f"{name}_{stage}"] = ev[stage][name]
            rows.append(row)
        self._append(self.pbt_log_path, self._pbt_log_fields, rows)

    # ------------------------------------------------------------------
    # segment selection per mode
    # ------------------------------------------------------------------

    def _uniform_segments(self, n):
        idx = self.rng.choice(len(self.pool), size=min(n, len(self.pool)), replace=False)
        return [(self.pool.segments[i].start, self.pool.segments[i].end) for i in idx]

    # ------------------------------------------------------------------
    # one generation
    # ------------------------------------------------------------------

    def _run_matches(self, gen):
        cfg = self.cfg
        self._match_rows = []
        # per-adversary tally of what it actually served this generation
        self._env_served = {i: {"n": 0, "regimes": []}
                            for i in range(len(self.env_pop))} if self.env_pop else {}

        for trader in self.traders:
            if self.env_pop is not None:
                k = min(cfg.matches_per_trader, len(self.env_pop))
                # same draw as before, so the random stream is unchanged
                opponent_idx = list(self.rng.choice(len(self.env_pop), size=k, replace=False))
            else:
                opponent_idx = [None] * cfg.matches_per_trader

            for m_idx, oi in enumerate(opponent_idx):
                opp = self.env_pop[oi] if oi is not None else None
                segs = (
                    opp.select_segments(self.pool, cfg.segments_per_match, self.rng)
                    if opp is not None
                    else self._uniform_segments(cfg.segments_per_match)
                )
                tp.train_on_segments(
                    trader, self.pool.df, self.feature_columns, segs,
                    self.env_kwargs, cfg.timesteps_per_match,
                )
                score, per_seg = tp.evaluate_on_segments(
                    trader, self.pool.df, self.feature_columns, segs,
                    self.env_kwargs, cfg.fitness_metric,
                )
                no_trade = [m["no_trade"] for m in per_seg]
                trader.record(score, float(np.mean(no_trade)))

                counted = True
                if opp is not None:
                    # all() -- a match is uninformative about the adversary only
                    # when the trader refused to trade on every segment it served
                    if cfg.env_skip_no_trade_matches and all(no_trade):
                        counted = False
                    else:
                        opp.record(self._adversary_score(segs, per_seg, score))
                    if counted:
                        self._env_served[oi]["n"] += 1
                        self._env_served[oi]["regimes"] += [
                            self._regime_by_start.get(st, "?") for st, _ in segs]

                self._match_rows.append(self._match_row(
                    gen, trader, m_idx,
                    "adversary" if opp is not None else "uniform",
                    oi, segs, score, per_seg, counted))

            # matches against archived adversaries: evaluation only, so the
            # trader cannot forget how to beat old opponents (anti-cycling)
            for h, entry in enumerate(self.env_hof.sample(cfg.hof_matches)):
                segs = entry.payload.select_segments(self.pool, cfg.segments_per_match, self.rng)
                score, per_seg = tp.evaluate_on_segments(
                    trader, self.pool.df, self.feature_columns, segs,
                    self.env_kwargs, cfg.fitness_metric,
                )
                trader.record(score, float(np.mean([m["no_trade"] for m in per_seg])))
                self._match_rows.append(self._match_row(
                    gen, trader, len(opponent_idx) + h, "hof", None,
                    segs, score, per_seg, False))

        # symmetric anti-cycling: adversaries are also scored against
        # archived trader snapshots, so an env individual can't look strong
        # merely by exploiting a quirk of the current generation's traders.
        # Snapshots are loaded into a dedicated shell network, never a live
        # trader (see _hof_shell).
        if self.env_pop is not None and len(self.trader_hof) > 0:
            shell = self._hof_shell
            for ind in self.env_pop:
                for entry in self.trader_hof.sample(cfg.env_hof_matches):
                    shell.model.set_parameters(entry.payload)
                    segs = ind.select_segments(self.pool, cfg.segments_per_match, self.rng)
                    score, per_seg = tp.evaluate_on_segments(
                        shell, self.pool.df, self.feature_columns, segs,
                        self.env_kwargs, cfg.fitness_metric,
                    )
                    ind.record(self._adversary_score(segs, per_seg, score))

    def _adversary_score(self, segs, per_seg, score):
        """What the adversary earns for a match it served (higher = harder)."""
        if self.cfg.env_fitness == "regret":
            return float(np.mean([
                self._ref_by_start[st] - m[self.cfg.fitness_metric]
                for (st, _), m in zip(segs, per_seg)
            ]))
        return -score

    def _external_eval(self, trader):
        _, per_segment = tp.evaluate_on_segments(
            trader, self.val_df, self.feature_columns, self.val_segments,
            self.env_kwargs, self.cfg.fitness_metric,
        )
        return per_segment[0]

    def _log_trader_external(self, gen, externals):
        """Held-out result for every trader, not just the best.

        generations.csv keeps reporting the best trader's external Sharpe, so
        that column is unchanged; this file adds the rest of the population.
        """
        ranked = sorted(self.traders, key=lambda t: t.fitness, reverse=True)
        rank_of = {id(t): r for r, t in enumerate(ranked)}
        self._append(self.ext_log_path, self._ext_log_fields, [
            {
                "generation": gen,
                "trader_id": t.trader_id,
                "rank": rank_of[id(t)],
                "fitness": t.fitness,
                "external_sharpe": e["sharpe"],
                "external_return_pct": e["total_return_pct"],
                "external_max_drawdown_pct": e["max_drawdown_pct"],
                "external_trades": e["trades"],
                "external_exposure": e["exposure"],
            }
            for t, e in zip(self.traders, externals)
        ])

    def run(self, on_generation=None):
        cfg = self.cfg
        for gen in range(cfg.generations):
            t_start = time.time()
            self._run_matches(gen)

            for trader in self.traders:
                trader.finalize_fitness()

            if self.env_pop is not None:
                for ind in self.env_pop:
                    ind.finalize_raw_fitness()
                ep.apply_fitness_sharing(
                    self.env_pop, self.pool.descriptor_matrix, cfg.sharing_strength)

            best_trader = max(self.traders, key=lambda t: t.fitness)
            # every trader on the held-out slice, so the internal-vs-external
            # relationship has n_traders points per generation instead of one
            externals = [self._external_eval(t) for t in self.traders]
            external = externals[self.traders.index(best_trader)]
            diversity = self._policy_diversity()

            self.trader_hof.add(
                snapshot_parameters(best_trader.model.get_parameters()),
                best_trader.fitness, gen)
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
                "external_val_sharpe_mean":
                    float(np.mean([e["sharpe"] for e in externals])),
                "external_val_sharpe_std":
                    float(np.std([e["sharpe"] for e in externals])),
                "trader_exposure_mean":
                    float(np.mean([e["exposure"] for e in externals])),
                "trader_no_trade_frac_mean":
                    float(np.mean([t.no_trade_frac for t in self.traders])),
                "policy_disagreement": diversity["policy_disagreement"],
                "identical_policy_pairs": diversity["identical_policy_pairs"],
                "clip_fraction_mean": float(np.nanmean(
                    [tp.ppo_diagnostics(t).get("clip_fraction", np.nan)
                     for t in self.traders])),
                "explained_variance_mean": float(np.nanmean(
                    [tp.ppo_diagnostics(t).get("explained_variance", np.nan)
                     for t in self.traders])),
                "seconds": round(time.time() - t_start, 1),
            }
            with open(self.log_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=self._log_fields).writerow(row)

            # parameter state as it stood *while earning* this generation's
            # fitness — logged before variation, so a row's parameters and its
            # fitness always refer to the same match results
            self._log_trader_params(gen)
            self._log_trader_external(gen, externals)
            self._append(self.match_log_path, self._match_log_fields, self._match_rows)
            if self.env_pop is not None:
                self._log_env_params(gen)

            if on_generation:
                on_generation(gen, row)

            # variation — after logging so the logged fitness matches this gen
            if self.env_pop is not None:
                self.env_pop = ep.evolve(
                    self.env_pop, self.rng,
                    elite_frac=cfg.elite_frac,
                    tournament_k=cfg.tournament_k,
                    weight_sigma=cfg.env_weight_sigma,
                    temp_sigma=cfg.env_temp_sigma,
                    fresh_frac=cfg.env_fresh_frac)
            if cfg.mode != "single":
                self._log_pbt_events(
                    gen,
                    tp.pbt_step(self.traders, self.rng, cfg.exploit_frac,
                                bounds=self.bounds),
                )

        best_trader = max(self.traders, key=lambda t: t.fitness)
        model_path = os.path.join(self.out_dir, "best_trader")
        best_trader.model.save(model_path)

        # The final-generation internal rank has ~zero correlation with
        # held-out performance, so "best_trader" is one draw from the final
        # population. Keep every final trader so evaluation can report the
        # population mean as well as the rank-0 pick.
        final_dir = os.path.join(self.out_dir, "final_traders")
        os.makedirs(final_dir, exist_ok=True)
        ranked = sorted(self.traders, key=lambda t: t.fitness, reverse=True)
        with open(os.path.join(final_dir, "index.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["trader_id", "rank", "fitness", "path"])
            w.writeheader()
            for rank, t in enumerate(ranked):
                p = os.path.join(final_dir, f"trader_{t.trader_id}")
                t.model.save(p)
                w.writerow({"trader_id": t.trader_id, "rank": rank,
                            "fitness": t.fitness, "path": f"trader_{t.trader_id}.zip"})
        return best_trader, model_path
