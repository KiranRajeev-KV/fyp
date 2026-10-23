# Team 100 - FYP

Competitive coevolutionary PPO for Bitcoin trading (CC-PPO): a population of
PPO trading agents is trained against an adversary population that chooses
which real historical BTC market windows the traders practise on. In the final
configuration, the adversary rewards changes in population performance on
previously examined windows (absolute learning progress).

## Reported final BTC development result

The final configuration reports mean per-segment Sharpe across six rolling
BTC evaluation folds (2022-03-23 to 2026-05-01, UTC). Each model's test window
is excluded from its own training; earlier test windows can enter later
folds' training data. The headline pools four six-seed configuration runs:
24 seed/configuration runs per arm and fold, using 12 distinct seed IDs
(42–53) with and without input drift. Agents pay perpetual-futures costs;
buy & hold uses spot costs.

| Strategy | Sharpe |
|---|---|
| Coevolution (this project) | 1.24 |
| Population-based training (PBT) | 1.24 |
| Buy & hold | 0.97 |
| Single PPO agent | 0.91 |
| Moving-average crossover | 0.15 |

Coevolution beats the single agent (P < 0.001) and the MA crossover
(P = 0.002), and ties PBT.

## Repo layout

- `ppo/` — all code: data pipeline, trading environment, coevolution engine,
  evaluation. See [`ppo/README.md`](ppo/README.md) for how to run it.
- `context.md` — design document.
- `docs/design-history.md` — how the architecture evolved across panel reviews.
- `docs/experiment-notes.md` — experiment notes and diagnoses.
- `docs/audit-2026-09.md` — September code audit and the fixes it led to.
- `download_klines.py` — downloads hourly BTCUSDT candles from the Binance public archive.

## Quick start

```bash
python3.12 -m venv ppo/.venv && source ppo/.venv/bin/activate
python -m pip install -r ppo/requirements-cpu.txt
python ppo/tests/smoke_test.py        # end-to-end check on synthetic data, no download needed
```
Preparing the BTC data and running the full evaluation are described in
[`ppo/README.md`](ppo/README.md).

### Members

- [Kiran Rajeev](https://github.com/KiranRajeev-KV)
- [Nandgopal R Nair](https://github.com/Nandgopal-R/)
- [Adarssh KG](https://github.com/adar-ssh)
- [Vijay S B](https://github.com/vijay-sb)
