# Design history

The project's architecture has changed twice since Panel Review 1. This note
exists so nobody picks up an older design's vocabulary or mechanism by
mistake — **`context.md` at the repo root is the current, authoritative
design.**

## PR1 (2026-02-23) — TensorNEAT-PPO hybrid
Proposed NEAT evolving network topology, with PPO doing gradient-based weight
optimization within each topology. Fitness via Sharpe/Sortino/Calmar. Nothing
implemented in code at this stage — literature survey and problem
formalization only.

## PR2 (2026-03-29) — Self-play CC-PPO
Pivoted to a single population of PPO agents competing against each other:
dominance-based fitness (fraction of opponents beaten), a Strategy Diversity
Index (mean KL divergence of action distributions vs. a reference
trajectory), adaptive opponent sampling (EMA win-rate, oversample hard
opponents), and MAP-Elites archiving.

**Status: superseded, not merged into the current design.** No code in this
repo implements the self-play loop, dominance fitness, SDI, adaptive opponent
sampling, or MAP-Elites — those numbers in the PR2 report (e.g. "PPO −4.34%
vs. Buy & Hold +6.78%", the 4-seed diversity-collapse experiment) were
produced outside the versioned codebase and aren't reproducible from what's
committed.

## Current design — environment-population coevolution (`context.md`)
Two **separate** populations with different operators:
- **Trader population**: PPO trading agents, evolved via PBT over
  hyperparameters (not weight-space crossover/mutation).
- **Environment population**: adversarial segment-selection strategies —
  small parameter vectors that pick which real historical windows a trader
  has to trade on. Restricted to *selecting* real segments, never inventing
  prices — the project's core, non-negotiable design constraint.

This is a materially different mechanism from PR2's self-play: there is no
opponent population of traders fighting each other, and no MAP-Elites
archive of trader behaviors. Instead, one side (traders) is optimized against
an adversarial other side (environment/segment selectors), with an external,
held-out evaluation harness guarding against Red Queen dynamics. See
`context.md` for the full design, milestone plan, and evaluation
non-negotiables.
