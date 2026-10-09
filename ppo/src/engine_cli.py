"""Shared command-line surface for EngineConfig.

Every driver that trains with CoevolutionEngine (run_coevolution.py,
run_walkforward.py) must build its config through here, so a walk-forward
run and a single-split run with the same flags are guaranteed to use the same
engine settings. Before this existed, run_walkforward.py silently used V1-era
defaults for every knob it did not expose.
"""

from coevo.engine import EngineConfig

_DEFAULT = EngineConfig()


def add_engine_args(parser):
    g = parser.add_argument_group("engine (EngineConfig)")
    g.add_argument("--generations", type=int, default=_DEFAULT.generations)
    g.add_argument("--n-traders", type=int, default=_DEFAULT.n_traders)
    g.add_argument("--n-env", type=int, default=_DEFAULT.n_env)
    g.add_argument("--timesteps-per-match", type=int, default=_DEFAULT.timesteps_per_match)
    g.add_argument("--matches-per-trader", type=int, default=_DEFAULT.matches_per_trader)
    g.add_argument("--segments-per-match", type=int, default=_DEFAULT.segments_per_match)
    g.add_argument("--hof-matches", type=int, default=_DEFAULT.hof_matches)
    g.add_argument("--env-hof-matches", type=int, default=_DEFAULT.env_hof_matches)
    g.add_argument("--hof-capacity", type=int, default=_DEFAULT.hof_capacity)
    g.add_argument("--fitness-metric", default=_DEFAULT.fitness_metric,
                   choices=["sharpe", "sortino", "calmar",
                            "total_return_pct", "max_drawdown_pct"])
    g.add_argument("--exploit-frac", type=float, default=_DEFAULT.exploit_frac)
    g.add_argument("--elite-frac", type=float, default=_DEFAULT.elite_frac)
    g.add_argument("--sharing-strength", type=float, default=_DEFAULT.sharing_strength)
    g.add_argument("--env-weight-sigma", type=float, default=_DEFAULT.env_weight_sigma,
                   help="adversary weight mutation scale")
    g.add_argument("--env-temp-sigma", type=float, default=_DEFAULT.env_temp_sigma,
                   help="adversary temperature mutation scale")
    g.add_argument("--env-fresh-frac", type=float, default=_DEFAULT.env_fresh_frac,
                   help="fraction of fresh random adversaries injected per generation")
    g.add_argument("--tournament-k", type=int, default=_DEFAULT.tournament_k)
    g.add_argument("--initial-balance", type=float, default=_DEFAULT.initial_balance)
    g.add_argument("--trading-fee", type=float, default=_DEFAULT.trading_fee)
    g.add_argument("--slippage", type=float, default=_DEFAULT.slippage)
    g.add_argument("--trade-penalty", type=float, default=_DEFAULT.trade_penalty,
                   help="anti-churn: extra reward penalty per executed trade")
    g.add_argument("--gamma", type=float, default=_DEFAULT.gamma,
                   help="PPO discount; raise to 0.995-0.999 for a longer horizon")
    g.add_argument("--ent-coef-max", type=float, default=_DEFAULT.ent_coef_max,
                   help="cap on evolved entropy coef")
    g.add_argument("--ent-coef-min", type=float, default=_DEFAULT.ent_coef_min,
                   help="floor on evolved entropy coef (try 0.005 to stop the "
                        "coevolution arm settling on hold-forever)")
    g.add_argument("--env-adversary-frac", type=float, default=_DEFAULT.env_adversary_frac,
                   help="share of each match's segments chosen by the adversary; "
                        "the rest are uniform draws (1.0 = original behaviour)")
    g.add_argument("--env-skip-no-trade-matches", action="store_true",
                   default=_DEFAULT.env_skip_no_trade_matches,
                   help="drop matches where the trader never traded from adversary fitness")
    g.add_argument("--env-fitness", default=_DEFAULT.env_fitness,
                   choices=["neg_sharpe", "regret"],
                   help="adversary objective: -trader score (original) or regret "
                        "vs the best trivial policy on each served segment")
    g.add_argument("--segment-sampling", default=_DEFAULT.segment_sampling,
                   choices=["uniform", "regime_balanced"],
                   help="how non-adversary segments are drawn; regime_balanced gives "
                        "every regime label the same share (oracle re-weighting on synthetic data)")
    g.add_argument("--shared-exam", type=int, default=_DEFAULT.shared_exam,
                   help="K > 0: rank traders on the same K uniform segments and credit each "
                        "adversary with the mean over all traders (0 = original)")
    g.add_argument("--overwrite", action="store_true", default=False,
                   help="allow replacing an out_dir that already holds a finished run")
    g.add_argument("--resume", action="store_true", default=False,
                   help="continue from out_dir/checkpoint/ if present (else start fresh)")
    return parser


ENGINE_FIELDS = [
    "generations", "n_traders", "n_env", "timesteps_per_match",
    "matches_per_trader", "segments_per_match", "hof_matches",
    "env_hof_matches", "hof_capacity", "fitness_metric", "exploit_frac",
    "elite_frac", "sharing_strength", "env_weight_sigma", "env_temp_sigma",
    "env_fresh_frac", "tournament_k", "initial_balance", "trading_fee",
    "slippage", "trade_penalty", "gamma", "ent_coef_max",
    "env_skip_no_trade_matches", "env_fitness", "overwrite", "resume",
    "ent_coef_min", "env_adversary_frac", "segment_sampling", "shared_exam",
]


def config_from_args(args, mode, seed):
    return EngineConfig(mode=mode, seed=seed,
                        **{f: getattr(args, f) for f in ENGINE_FIELDS})
