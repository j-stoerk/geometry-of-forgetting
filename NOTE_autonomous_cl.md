# Boundary-free, autonomous continual learning

Closes the paper's most load-bearing operating assumption -- that task boundaries are
GIVEN.  Validation: `evaluate_autonomous_cl.py`; reusable class:
`AutonomousController` in `interference_ledger.py`.

## The loop (no task labels, no resets)
Per incoming batch of feature rows, using only geometry the ledger already tracks:
1. **Novelty** = fraction of the batch's energy outside the current working subspace.
2. **Novelty persistence** -- a novel batch is HELD; if novelty persists K steps it is a
   genuine new task (consolidate the old task as a protected anchor, then adapt the new
   one); if it subsides it was an OOD blip (dropped, never adapted).  This is what
   distinguishes a task shift from noise -- both look novel for one batch.
3. **Working-subspace drift** away from the last consolidation catches GRADUAL shifts
   that never spike per-batch novelty.
4. **Gated update** projects the step off the consolidated anchor subspaces (the QP gate).
5. **Bounded cost** -- anchors are capped and the closest pair is Sigma-merged when full,
   so per-step cost is independent of stream length.
Scale-normalized tracking + per-step clipping make it robust to large OOD outliers.

## Result (12 seeds, vs an oracle GIVEN the true boundaries)
| regime  | boundary F1 | naive | oracle | autonomous |
|---------|-------------|-------|--------|------------|
| clean   | 1.00        | 0.607 | 0.001  | **0.001**  |
| noisy   | 0.96        | 0.722 | 0.022  | **0.003**  |
| gradual | 0.81        | 0.815 | 0.001  | **0.274**  |

- **Clean shifts: matches the oracle exactly** -- with no boundaries given, retention is
  identical to knowing them (F1 1.00).
- **Noisy: matches/beats the oracle** (F1 0.96) -- it skips the OOD batches the oracle
  naively adapts on.
- **Gradual: degrades gracefully** (F1 0.81, ~2/3 of the oracle gap recovered) -- the
  inherently hard case, detected with correct latency (you cannot detect a smooth shift
  until it accumulates).

## What this does and does not close
Closes: the "boundaries assumed" limitation, in the exact/frozen-feature regime, with
bounded per-step cost (also touching the O(T) constraint-growth gap via anchor capping).
Still open: validation on a real unlabeled LLM stream (not just synthetic drift), and the
gradual case is approximate, not exact.
