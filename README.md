# Team 100 - FYP

Competitive coevolutionary PPO for Bitcoin trading (CC-PPO): a population of
PPO trading agents is trained against an adversary population that chooses
which real historical BTC market windows the traders practise on. The
adversary rewards windows where the traders are still learning (learning
progress), so training keeps moving to the market conditions the traders
handle worst.

## Result

Scored on six BTC periods (Mar 2022 – May 2026) that no agent trained on,
24 seeds, realistic perpetual-futures costs:

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
cd ppo
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd tests && python smoke_test.py        # end-to-end check on synthetic data, no download needed
```
Preparing the BTC data and running the full evaluation are described in
[`ppo/README.md`](ppo/README.md).

### Members
- [Kiran Rajeev](https://github.com/KiranRajeev-KV)
- [Nandgopal R Nair](https://github.com/Nandgopal-R/)
- [Adarssh KG](https://github.com/adar-ssh)
- [Vijay S B](https://github.com/vijay-sb)
