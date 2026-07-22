"""Hall of fame archives for both populations — the anti-cycling guardrail.

Traders are also evaluated against archived environment individuals, so the
population cannot "forget" how to handle old adversaries; environment
individuals are scored against snapshots of past traders analogously.
Trader snapshots are stored as SB3 parameter dicts (cheap, in-memory for
small MLP policies).
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class HofEntry:
    payload: object     # EnvIndividual for env HOF; parameter dict for traders
    fitness: float
    generation: int


class HallOfFame:
    def __init__(self, capacity=20, rng=None):
        self.capacity = capacity
        self.entries = []
        self.rng = rng or np.random.default_rng()

    def add(self, payload, fitness, generation):
        self.entries.append(HofEntry(payload, fitness, generation))
        if len(self.entries) > self.capacity:
            # keep the strongest, but evict randomly among the rest so the
            # archive retains historical diversity, not just one era's elite
            self.entries.sort(key=lambda e: e.fitness, reverse=True)
            protected = self.entries[: self.capacity // 2]
            rest = self.entries[self.capacity // 2:]
            drop = int(self.rng.integers(0, len(rest)))
            rest.pop(drop)
            self.entries = protected + rest

    def sample(self, k):
        if not self.entries:
            return []
        k = min(k, len(self.entries))
        idx = self.rng.choice(len(self.entries), size=k, replace=False)
        return [self.entries[i] for i in idx]

    def __len__(self):
        return len(self.entries)
