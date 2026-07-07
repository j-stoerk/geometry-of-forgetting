# Toward scalable, true continual learning: the synthesis

True continual learning needs three properties at once, each solved separately in this
repository; the open problem was that they *compose* into one bounded-cost learner
(`evaluate_scalable_cl.py`).

| property | mechanism | note |
|---|---|---|
| boundary-free | memory-metric flow `θ̇ = −(M+εI)⁻¹∇L` on a reservoir metric | fresh dirs (small M) learn, buffered (past) dirs damped -- boundaries emerge |
| drift-robust | protect the CURRENT function geometry (I1 / transport I2) | the backbone moves, so a stored metric goes stale |
| bounded | low-rank M (Woodbury, O(dr)) + capped anchor reservoir | cost independent of stream length |

**The tension and its resolution.** Drift-robustness seems to require recomputing every
past task's Jacobian every step (O(stream)).  *Transport* resolves it: refresh the
low-rank metric from the bounded anchor set every `τ` steps -- O(1) in stream length --
which tracks the drift without per-step, per-task backprop.  Because the metric is built
from the anchor *reservoir*, recent tasks enter it with a delay: learnable while fresh,
protected once buffered -- boundary handling for free.

**Result** (2-layer net, backbone drifts, unlabeled continuous stream; retention on past
regimes, lower is better -- see script output for the current numbers):
- naive forgets; **stale** (frozen metric) drifts out of date; **functional** (refresh
  every step) is drift-proof but O(stream); **transported** (refresh every `τ`) recovers
  most of functional's retention gain at a small fraction of the refresh cost.
- Cost scaling: functional's refresh count grows with the number of regimes; transported
  stays bounded per unit stream regardless of how many regimes have passed.

So boundary-free + drift-robust + bounded compose into a single learner whose per-step
protection cost is O(1) in stream length -- the shape of a scalable, true continual
learner.  Open: a deep-network / real-stream instantiation (the metric is a running
Kronecker-factored second moment; transport is a per-layer activation-drift map measured
on the anchor reservoir).
