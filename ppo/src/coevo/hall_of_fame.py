"""Hall of fame archives for both populations — the anti-cycling guardrail.

Traders are also evaluated against archived environment individuals, so the
population cannot "forget" how to handle old adversaries; environment
individuals are scored against snapshots of past traders analogously.
Trader snapshots are stored as SB3 parameter dicts (cheap, in-memory for
small MLP policies).
"""

import copy
from dataclasses import dataclass

import numpy as np


@dataclass
class HofEntry:
    payload: object     # EnvIndividual for env HOF; parameter dict for traders
    fitness: float
    generation: int


def snapshot_parameters(params):
    """Deep-copy an SB3 ``get_parameters()`` dict so the archive is a real snapshot.

    ``get_parameters()`` returns ``state_dict()``s whose tensors SHARE STORAGE
    with the live network. Archiving them directly means the "snapshot" keeps
    tracking the live trader (including after PBT overwrites it), and any
    ``set_parameters(entry)`` on a shell model that borrows a live trader's
    tensors clobbers that trader. Cloning to CPU breaks both links.
    """
    out = {}
    for key, value in params.items():
        if isinstance(value, dict):
            out[key] = {
                name: (t.detach().clone().cpu() if hasattr(t, "detach") else copy.deepcopy(t))
                for name, t in value.items()
            }
        else:
            out[key] = copy.deepcopy(value)
    return out


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
