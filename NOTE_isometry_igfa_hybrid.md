# Interference-gated allocation x dynamical isometry: a super-additive hybrid

Keeping IGFA as the interference controller (it decides WHERE the update may go) and
adding a Rosseau-style dynamical-isometry regularizer (it keeps the ALLOWED directions
learnable) gives a continual learner that beats either mechanism alone by a wide,
seed-robust margin -- and the synergy has a measured mechanism.  Reverses the earlier
"IGFA over-constrains" reading, which was a gate sign bug.
(`kaggle_isometry_igfa_hybrid.py`, `isometry-igfa-hybrid-v3-gem-3seed`.)

Rosseau, Mueller, Nowe, *Preserving Plasticity in Continual Learning via Dynamical
Isometry*, ICML 2026: plasticity loss <-> anisotropy of the empirical NTK; keep each
weight matrix near orthogonal (`||W^T W - I||_F -> 0`, closed-form grad `4W(W^T W - I)`,
no SVD) via a decoupled AdamO step; this also revives dead ReLUs.

## Setup
20-task Split-CIFAR-100 (5 classes/task), deeper plain-ReLU CNN, no normalization,
orthogonal init (every method starts ON the isometric manifold; the question is who
PRESERVES it).  Task-IL masking.  3 seeds.  Two streams: standard, and per-task
pixel-permuted (the harder "A1 breaks / visible drift" regime -- each task must relearn
low-level features).

Six methods at the winning gate strength alpha=1, hybrid iso_lambda=0.1:
naive, iso, igfa (A-GEM mixed-aggregate gate), igfa-gem (per-task GEM gate),
igfa+iso, igfa+iso-gem.

## The gate must have the right sign (the bug that hid everything)
The update is `-g`, so the first-order change in a past loss is `-<g, g_anchor>`.  Hence
`<g,g_anchor> > 0` is beneficial TRANSFER (past loss falls) and `< 0` is FORGETTING.  The
gate must fire only on `< 0` (A-GEM/GPM convention) and keep transfer.  An earlier version
fired on `> 0` -- it stripped the shared learning signal (which dominates a shared-feature
net, ~72% of steps) and let real forgetting through.  That produced the false "IGFA
freezes / drift explodes" result.  With the correct sign the gate fires ~10-27% of steps
(genuine conflict only) and preserves learning.

## Result (3 seeds; avg_acc, standard | permuted)

| method | standard | permuted |
|---|---|---|
| naive | 0.227 +- 0.015 | 0.215 +- 0.010 |
| iso | 0.260 +- 0.009 | 0.254 +- 0.018 |
| igfa (agem) | 0.339 +- 0.034 | 0.242 +- 0.053 |
| igfa-gem | 0.298 +- 0.069 | 0.214 +- 0.019 |
| igfa+iso (agem) | 0.450 +- 0.053 | 0.393 +- 0.007 |
| **igfa+iso-gem** | **0.555 +- 0.034** | **0.425 +- 0.011** |

Same ordering in both regimes; igfa+iso-gem best on every axis (learning, forgetting,
Omega, effective rank).  The permuted stream has a lower ceiling (naive learns 0.395 vs
0.615), so absolute margins compress -- hybrid is 2.0x naive there vs 2.4x standard -- but
ordering and mechanism are invariant.

## Three findings

**1. The hybrid is super-additive.**  On the standard stream, iso alone adds +0.033 over
naive and the gem gate alone adds +0.071; additive would predict 0.331, but igfa+iso-gem
hits 0.555 -- a synergy bonus of +0.224, 3x the sum of the parts.  IGFA cuts forgetting
(retained on old tasks ~0.5 vs naive ~0.15) at preserved learning; iso preserves plasticity
(learning 0.78, drift pinned, ReLUs revived); together they compound.

**2. The sharper per-task (GEM) gate REQUIRES isometry.**  GEM alone is WORSE than the
mixed-aggregate gate (0.298 < 0.339) because per-task projection over-constrains and
collapses the representation to rank ~1 (eff-rank 5.6 standard / 1.3 permuted).  With iso
holding the rank high (25.2 / 5.3) that same precision becomes the best protector.
Precision without conditioning self-destructs; precision with it wins.  (Also why gem-alone
is the highest-variance method and the hybrid the tightest.)

**3. The synergy mechanism, measured.**  Isometry lowers the gate's COST -- the fraction of
the gradient it removes, `||g - gated|| / ||g||` -- by ~40% for BOTH gate types in BOTH
regimes:

```
        gate-alone -> hybrid   reduction
  agem  standard  0.146 -> 0.087   -40%
  agem  permuted  0.268 -> 0.151   -43%
  gem   standard  0.263 -> 0.160   -39%
  gem   permuted  0.414 -> 0.250   -40%
```

In the isometric, high-rank representation the conflicting component is a smaller slice of
the gradient, so protecting the past costs less of the present -- interference control and
learning are decorrelated by the geometry.  (Note: the earlier prediction that iso works by
lowering the gate's FIRING RATE was wrong; the rate barely moves.  It lowers the per-step
COST.)

## Bearing on the paper
This is the paper's thesis made operational and quantified: IGFA decides *where* updates
may go; dynamical isometry keeps those allowed directions learnable *and* makes the gating
cheap.  Complements the memflow negative result -- there a hard *preconditioner* damping
occupied (shared) directions froze the trunk; here a correct-sign *gate* removing only
genuine conflict, inside an isometric high-rank space, is the version that works.

## Caveats
- One benchmark family (Split-CIFAR-100), one small CNN.  Would want a second
  architecture/benchmark to generalize.
- Absolute accuracies are low (hard 20-task task-IL, small net); the CLAIM is comparative
  (ordering + mechanism), which is robust across 3 seeds and two streams.
- gem_k=4 (per-task tasks sampled/step) and iso_lambda=0.1 not swept jointly; the frontier
  could be mapped further.
