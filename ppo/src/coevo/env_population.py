"""Environment population (the adversary).

Each individual is a segment-selection strategy: preference weights over the
segment descriptor features plus a temperature. It scores every segment in
the pool as descriptor . weights and samples segments through a softmax —
it can only choose among real historical windows, never invent prices.

Fitness is the inverted trader score on the segments it served, with
competitive fitness sharing so individuals are rewarded for being difficult
in *different* ways (guards against diversity collapse).
"""

from dataclasses import dataclass, field

import numpy as np

from segments import DESCRIPTOR_NAMES

WEIGHT_BOUND = 3.0
TEMP_BOUNDS = (0.2, 3.0)


@dataclass
class EnvIndividual:
    weights: np.ndarray          # preference over descriptor features
    temperature: float
    fitness: float = 0.0
    raw_fitness: float = 0.0
    _scores: list = field(default_factory=list)

    def selection_probs(self, descriptor_matrix):
        logits = descriptor_matrix @ self.weights / self.temperature
        logits -= logits.max()
        p = np.exp(logits)
        return p / p.sum()

    def select_segments(self, pool, n, rng):
        probs = self.selection_probs(pool.descriptor_matrix)
        n = min(n, len(pool))
        idx = rng.choice(len(pool), size=n, replace=False, p=probs)
        return [(pool.segments[i].start, pool.segments[i].end) for i in idx]

    def record(self, trader_score):
        # adversary wants the trader to do badly
        self._scores.append(-trader_score)

    def finalize_raw_fitness(self):
        self.raw_fitness = float(np.mean(self._scores)) if self._scores else 0.0
        self._scores = []


def random_individual(rng):
    return EnvIndividual(
        weights=rng.uniform(-1.0, 1.0, size=len(DESCRIPTOR_NAMES)),
        temperature=float(rng.uniform(0.5, 1.5)),
    )


def crossover(a, b, rng):
    alpha = rng.uniform(0.0, 1.0, size=len(a.weights))
    weights = alpha * a.weights + (1 - alpha) * b.weights
    temperature = float(np.exp(
        rng.uniform() * np.log(a.temperature) +
        (1 - rng.uniform()) * np.log(b.temperature)
    ))
    return EnvIndividual(weights=weights, temperature=temperature)


def mutate(ind, rng, weight_sigma=0.3, temp_sigma=0.15):
    ind.weights = np.clip(
        ind.weights + rng.normal(0, weight_sigma, size=ind.weights.shape),
        -WEIGHT_BOUND, WEIGHT_BOUND,
    )
    ind.temperature = float(np.clip(
        ind.temperature * np.exp(rng.normal(0, temp_sigma)),
        *TEMP_BOUNDS,
    ))
    return ind


def apply_fitness_sharing(population, descriptor_matrix, sharing_strength=1.0):
    """Penalize individuals whose selection distributions overlap.

    Similarity between two individuals is the Bhattacharyya coefficient of
    their softmax selection distributions over the pool. Shared fitness =
    raw / (1 + strength * mean similarity to others).
    """
    probs = [ind.selection_probs(descriptor_matrix) for ind in population]
    n = len(population)
    for i, ind in enumerate(population):
        if n == 1:
            ind.fitness = ind.raw_fitness
            continue
        sims = [
            float(np.sum(np.sqrt(probs[i] * probs[j])))
            for j in range(n) if j != i
        ]
        niche = 1.0 + sharing_strength * float(np.mean(sims))
        # shift-invariant sharing: divide the advantage, not the raw value
        ind.fitness = ind.raw_fitness / niche if ind.raw_fitness > 0 else ind.raw_fitness * niche


def evolve(population, rng, elite_frac=0.25, tournament_k=3):
    """Truncation elitism + tournament-selected crossover/mutation offspring."""
    ranked = sorted(population, key=lambda ind: ind.fitness, reverse=True)
    n_elite = max(1, int(len(ranked) * elite_frac))
    elites = [
        EnvIndividual(weights=e.weights.copy(), temperature=e.temperature)
        for e in ranked[:n_elite]
    ]

    def tournament():
        contenders = rng.choice(len(ranked), size=min(tournament_k, len(ranked)), replace=False)
        return ranked[min(contenders)]

    offspring = []
    while len(elites) + len(offspring) < len(population):
        child = crossover(tournament(), tournament(), rng)
        offspring.append(mutate(child, rng))

    return elites + offspring
