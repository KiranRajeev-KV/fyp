# Team 100 - FYP

Competitive coevolutionary PPO for cryptocurrency trading (CC-PPO): a
population of PPO trading agents evolved against an adversarial population
that selects which real historical market segments the traders have to
trade on. The hypothesis under test is generalization, not raw profit — see
[`context.md`](context.md) for the full design, evaluation standards, and
milestone plan, and [`docs/design-history.md`](docs/design-history.md) for
how the architecture arrived here from earlier panel reviews.

## Status

The full pipeline is implemented in `ppo/`: data pipeline and feature
engineering, regime labeling, segment pool + walk-forward splitting, a gym
environment with fees and slippage, the coevolutionary engine (trader
population with PBT, adversarial environment population, halls of fame,
fitness sharing, external-yardstick logging), baselines, and a per-regime
evaluation harness with bootstrap significance. Verified end-to-end on
synthetic data (`ppo/tests/smoke_test.py`); full-scale experiments on the
real BTCUSDT data (three-way ablation, multi-seed) are the next step.
See `ppo/README.md` for how to run everything.

## Repo layout

- `context.md` — current design doc (read this first).
- `docs/design-history.md` — how the architecture evolved across panel reviews.
- `ppo/` — data pipeline, gym environment, and PPO baseline code.
- `reviews/` — panel review reports and slides (PR1, PR2).

### Members
[Kiran Rajeev](https://github.com/KiranRajeev-KV)
