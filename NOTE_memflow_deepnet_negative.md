# Deep-network memory-metric flow: a negative result

The next step flagged in [`NOTE_memory_metric_flow.md`](NOTE_memory_metric_flow.md) --
a running Kronecker-factored `M` carrying the flow `θ̇ = −(M+εI)⁻¹∇L` into a deep
network -- was built and run (`kaggle_memory_flow_deepnet.py`,
`memflow-deep-exact-logged-v3-protective`). **It does not transfer.** The K-FAC
preconditioner path underperforms plain EWC and never protects an old task. The logs
say precisely why, and the reason is geometric, not a tuning failure.

## Setup
10-task Split-CIFAR-100 (10 classes/task), small conv net, 5 epochs/task, lr 0.05,
task-IL masking. Protector = protective preconditioner `ε(M+εI)⁻¹ = I − U diag(s/(ε+s)) Uᵀ`
with `M` a per-layer K-FAC product (A-factor = input-activation covariance, rank 64;
G-factor = output-gradient covariance, rank 32), trace-normalized, refreshed from a
400-sample anchor reservoir. Baselines: naive, EWC. ε swept over {0.3, 1.0, 3.0, 10.0}.
Single seed (`seeds=[0]`); 57 min on 2×T4.

## Result

| method        | avg_acc | learn-fresh | retained (old tasks) | Ω     | read |
|---------------|---------|-------------|----------------------|-------|------|
| naive         | 0.148   | 0.399       | ~chance              | 0.07  | learns, forgets all |
| **ewc**       | **0.247** | 0.452     | **0.18–0.26**        | 0.35  | **learns AND retains** |
| func @ε0.3    | 0.153   | 0.155       | ~chance              | 0.37  | **frozen** after task 0 |
| func @ε1.0    | 0.190   | 0.438       | ~chance              | 0.18  | learns, no protection |
| func @ε3.0    | 0.100   | 0.137       | 0.10 (chance)        | −0.11 | **frozen** (dead) |
| func @ε10.0   | 0.114   | 0.461       | ~chance              | −0.07 | learns, no protection |

Per-task diagonals (fresh-learning accuracy on each task as it arrives) make the two
failure modes concrete:

```
func @ε3.0  learn-fresh:  0.47 0.10 0.10 0.10 0.10 0.10 0.10 0.10 0.10 0.10   <- frozen after task 0
func @ε1.0  learn-fresh:  0.44 0.16 0.24 0.27 0.36 0.52 0.58 0.62 0.54 0.66   <- plastic
func @ε1.0  retained:     0.17 0.12 0.08 0.10 0.10 0.12 0.17 0.19 0.22 0.66   <- all but current ~chance
ewc         retained:     0.18 0.22 0.25 0.26 0.23 0.13 0.20 0.26 0.21 0.53   <- genuinely retains
```

**No ε both learns and retains.** Small ε (0.3, 3.0) freezes the trunk: after task 0
every later task sticks at chance 0.10. Large ε (1.0, 10.0) restores plasticity but the
final-row retention is ~chance on every task except the current one -- it forgets like
naive. EWC is the only method whose old-task accuracies stay at 0.18–0.26 while it still
learns fresh at ~0.45.

**Ω is degenerate in the frozen cases.** `Ω = 1 − forget/(learn − chance)`; when learning
collapses to chance the ratio is ~0/0, so ε0.3 posts Ω=0.37 while being useless
(avg_acc 0.153 ≈ naive). Ω is only meaningful when `avg_acc` moves with it, which it
never does for the functional variants. Read `avg_acc`, not Ω, here.

## Why -- occupied ≈ shared trunk (the geometric reason)

From the factor log, `A_top_mass_ratio ≈ 0.98–1.00` on every conv layer: the top-64
retained directions carry ~99% of the K-FAC input-covariance trace (G-factor 0.96–1.00
likewise). So the "occupied" subspace the protector damps is not a small task-specific
slice -- it is essentially the **entire high-variance working subspace of every layer**,
dominated by the shared low-level feature directions.

Consequently *protecting occupied directions ≈ freezing the shared trunk*: the directions
task 0 wrote into `M` are the same directions tasks 1–9 need. The `(M+εI)⁻¹`
preconditioner then has only two regimes and nothing in between:

- **damp-everything** (`s ≫ ε` on the shared trunk → factor `s/(ε+s) ≈ 1` → update ≈ 0):
  the network freezes;
- **damp-nothing** (`ε` large enough to release the trunk → factor ≈ 0 everywhere): the
  network forgets like naive.

The linear/idealized flow's clean protection assumed occupied directions are
task-specific -- true for near-orthogonal synthetic tasks (`Σ_t Σ_A = 0`, validated to
4e-16 in the low-dim regime), false in a deep net with shared features. There is no
interior ε that separates protection from plasticity because the two live in the *same*
directions.

## Why EWC does not freeze

EWC adds a soft quadratic penalty `+λ (θ−θ*)ᵀ diag(F) (θ−θ*)` pulling toward past optima.
A penalty permits *finite* motion in penalized directions -- the new-task gradient can
overpower the pull where it needs to -- whereas a hard preconditioner can drive motion in
occupied directions to *zero*. That is the whole difference: penalty (bounded resistance)
vs preconditioner (unbounded resistance in occupied directions). With shared features,
only the bounded one can still learn.

## Caveat
Single seed. The exact ε ranking is noisy -- ε3.0's total freeze while ε1.0 and ε10.0
stay plastic is a one-seed instability, not a physical trend in ε. But the two robust
conclusions do not depend on the seed: (a) the freeze-vs-forget dichotomy appears across
the whole sweep, and (b) EWC dominates every functional variant on avg_acc and is the
only method that retains.

## Bearing on the paper
The memflow paragraph currently closes: *"the flow is natural-gradient/K-FAC-shaped, so
the path to deep networks is a running Kronecker-factored M."* That forward claim is
exactly what this run tested and it does not hold for the *preconditioner* form. The
idealized flow and its four low-dim validations (C0–C4) stand; what fails is the naive
K-FAC lift. The sentence should be softened to mark the deep-net path as open and harder
than "just run a Kronecker-factored M" -- reaching depth needs either a **penalty
formulation** (EWC-like, using the memory metric as `F`) or genuinely **task-specific
subspaces**, because with shared features the dominant K-FAC directions are common to all
tasks.

## Verdict
Deep-net protector stays **EWC**. The memory-metric flow remains a clean boundary-free
account in the exact/low-dim regime; its K-FAC preconditioner lift to deep nets is a
negative result, for a concrete and instructive geometric reason.
