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
    shift: float = 0.0           # added to the perturbed inputs (EngineConfig.obs_adversary)
    log_scale: float = 0.0       # their scale is multiplied by exp(log_scale)
    fitness: float = 0.0
    raw_fitness: float = 0.0
    # diagnostics filled in each generation, logged so the adversary side of
    # the arms race can be read parameter-by-parameter
    niche: float = 1.0           # 1 + strength * mean overlap with the others
    origin: str = "random"       # random | elite | offspring | fresh
    # Did this individual actually get tested this generation? With 8 traders
    # drawing 3 opponents each from 16, some are never drawn. An untested
    # individual has no evidence for or against it, so it must not be eligible
    # to be copied forward as an elite -- see evolve().
    evaluated: bool = False
    _scores: list = field(default_factory=list)

    def selection_entropy(self, descriptor_matrix):
        """Shannon entropy (nats) of the selection distribution.

        Low = this adversary always serves the same few windows (specialised,
        and a candidate for the diversity penalty); high = near-uniform, i.e.
        it has stopped choosing and behaves like the random-lottery baseline.
        """
        p = self.selection_probs(descriptor_matrix)
        return float(-np.sum(p * np.log(p + 1e-12)))

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

    def record(self, adversary_score):
        # Higher = the match was harder for the trader. The engine decides how
        # that is measured (EngineConfig.env_fitness): -trader_score for the
        # original objective, or regret against the best trivial policy.
        self._scores.append(float(adversary_score))

    def finalize_raw_fitness(self):
        # raw_fitness stays 0.0 for an untested individual only so the sorting
        # keys stay numeric; `evaluated` is what selection actually consults.
        self.evaluated = bool(self._scores)
        self.raw_fitness = float(np.mean(self._scores)) if self._scores else 0.0
        self._scores = []


def random_individual(rng):
    return EnvIndividual(
        weights=rng.uniform(-1.0, 1.0, size=len(DESCRIPTOR_NAMES)),
        temperature=float(rng.uniform(0.5, 1.5)),
        shift=float(rng.uniform(-1.0, 1.0)),
        log_scale=float(rng.uniform(-0.5, 0.5)),
    )


def crossover(a, b, rng):
    alpha = rng.uniform(0.0, 1.0, size=len(a.weights))
    weights = alpha * a.weights + (1 - alpha) * b.weights
    temperature = float(np.exp(
        rng.uniform() * np.log(a.temperature) +
        (1 - rng.uniform()) * np.log(b.temperature)
    ))
    u = rng.uniform()
    return EnvIndividual(weights=weights, temperature=temperature,
                         shift=u * a.shift + (1 - u) * b.shift,
                         log_scale=u * a.log_scale + (1 - u) * b.log_scale)


def mutate(ind, rng, weight_sigma=0.3, temp_sigma=0.15):
    ind.weights = np.clip(
        ind.weights + rng.normal(0, weight_sigma, size=ind.weights.shape),
        -WEIGHT_BOUND, WEIGHT_BOUND,
    )
    ind.shift = float(np.clip(ind.shift + rng.normal(0, weight_sigma), -2.0, 2.0))
    ind.log_scale = float(np.clip(ind.log_scale + rng.normal(0, temp_sigma), -1.0, 1.0))
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
        ind.niche = niche
        # shift-invariant sharing: divide the advantage, not the raw value
        ind.fitness = ind.raw_fitness / niche if ind.raw_fitness > 0 else ind.raw_fitness * niche


def evolve(population, rng, elite_frac=0.25, tournament_k=3,
           weight_sigma=0.3, temp_sigma=0.15, fresh_frac=0.0):
    """Truncation elitism + tournament-selected crossover/mutation offspring.

    weight_sigma / temp_sigma set the adversary variation rate — raise them to
    keep the environment population engaged (mitigates disengagement, where the
    adversary is outrun by the trader and its fitness collapses).

    fresh_frac injects that fraction of brand-new random individuals each
    generation ("fresh blood"). Once the population converges on the few segment
    types a strong trader has already mastered, its fitness collapses; fresh
    random adversaries keep probing unexplored regions of the segment pool,
    slowing that collapse. 0.0 = disabled (original behavior).
    """
    ranked = sorted(population, key=lambda ind: ind.fitness, reverse=True)
    n = len(ranked)
    n_elite = max(1, int(n * elite_frac))
    n_fresh = int(n * fresh_frac)

    # Only individuals that were actually tested may be carried forward or act
    # as parents. Without this an individual that was never drawn as an
    # opponent sits at raw_fitness 0.0, which outranks the typically-negative
    # scores of individuals that did play, and gets copied forward as an elite
    # on no evidence at all. Fall back to the full ranking in the degenerate
    # case where nothing was tested.
    eligible = [ind for ind in ranked if ind.evaluated] or ranked
    n_elite = min(n_elite, len(eligible))

    elites = [
        EnvIndividual(weights=e.weights.copy(), temperature=e.temperature,
                      shift=e.shift, log_scale=e.log_scale,
                      origin="elite")
        for e in eligible[:n_elite]
    ]

    def tournament():
        contenders = rng.choice(len(eligible), size=min(tournament_k, len(eligible)), replace=False)
        return eligible[min(contenders)]

    offspring = []
    while len(elites) + len(offspring) + n_fresh < n:
        child = crossover(tournament(), tournament(), rng)
        child.origin = "offspring"
        offspring.append(mutate(child, rng, weight_sigma=weight_sigma, temp_sigma=temp_sigma))

    fresh = []
    for _ in range(n_fresh):
        ind = random_individual(rng)
        ind.origin = "fresh"
        fresh.append(ind)
    return elites + offspring + fresh
