# Boundary-free continual learning as a memory-metric flow

A differential-geometric derivation of autonomous continual learning: no tasks, no
boundaries, no detection thresholds -- one accumulating metric and one flow.  Validated
in `evaluate_memory_metric_flow.py`.  (Conservation-law + reduced-order structure-
preserving submanifold + metric-preserving flow, after Friedl et al., *Learning
Hamiltonian Dynamics at Scale*, 2026.)

## The reframe
Discrete tasks with boundaries are replaced by a continuous data path `D_t` inducing an
instantaneous Gauss--Newton metric `G_t = E_{x~D_t}[∇f ∇fᵀ]`.  Accumulate it into a
**leaky memory metric**

```
    dM/dt = G_t − λ M_t ,        M_0 = 0,
```

and run the **memory-natural-gradient flow**

```
    θ̇ = −(M_t + εI)⁻¹ ∇L_t .
```

`M_t` is the exponentially-weighted geometry the learner has seen; its eigenstructure is
the occupied subspace, continuous and graded.  Fresh directions (small M-eigenvalue)
have effective learning rate `≈ lr` (learn like naive); occupied directions (eigenvalue
`s`) are damped by `ε/(ε+s)` (protected).  The hard projection gate `(I−QQᵀ)` is the
`ε→0`, spectral-hard limit -- so this is the boundary-free, graded generalization of the
paper's gate.

## The adiabatic-retention law
For any past slice `A` with quadratic loss `L_A = ½(θ−θ_A*)ᵀΣ_A(θ−θ_A*)`, the chain rule
gives, along the flow, exactly

```
    dL_A/dt = −∇L_Aᵀ (M_t + εI)⁻¹ ∇L_t .
```

- **Fresh current gradient** (`Σ_t Σ_A = 0`, and `A` written into `M` orthogonally):
  `dL_A/dt = 0` **exactly** -- past loss is conserved (validated to 4e-16).  Past-task
  losses are therefore *adiabatic invariants* of the slow learning flow.
- **Conflicting current gradient**: the leak is the interference measured in the `M⁻¹`
  inner product -- the continuous generalization of the interference energy `½ΔᵀΣΔ`.

No boundary is detected; a task shift is just `G_t`'s support moving to fresh directions,
which grows a new low-eigenvalue block of `M` that the flow learns freely while the old
blocks (now high-eigenvalue) are damped.

## Validation (numpy, `evaluate_memory_metric_flow.py`)
- **C0** the law is exact: fresh leak `4e-16`, conflicting leak `= −∇L_Aᵀ(M+εI)⁻¹∇L_t`.
- **C1** adiabatic retention: the flow drives a probe's loss down (`0.52→0.05`) then
  **holds it flat** (`~0.08`) through five later regimes; naive learns it (`→0.00`) then
  loses it (`→0.63`).  No boundaries used.
- **C2** boundary-free vs naive vs hard-gate (10 seeds): the flow beats naive on drift
  (`0.25` vs `0.76`), jump (`0.12` vs `0.22`), and noisy (`0.26` vs `0.94`) *uniformly*
  -- one mechanism, no per-regime special-casing -- and beats hard-gating every past
  subspace (`1.67`), which over-constrains overlapping tasks the soft flow trades off.
- **C3** `λ` is the retention--plasticity knob: plasticity rises monotonically with `λ`
  (`0.44→0.04`); retention has an **interior optimum** (`λ≈0.05`) where the memory
  timescale `1/λ` matches the drift rate.  One principled parameter sets the frontier --
  addressing the plasticity gap directly.
- **C4** reduced-order `O(dr)` metric (top-r eigen + Woodbury inverse) matches the full
  `O(d³)` flow (`|diff| 0.02`) -- the low-rank structure-preserving submanifold carries
  the whole flow (the Friedl reduced-order idea).

## What this supersedes and what stays open
Supersedes the heuristic autonomous controller (novelty/drift thresholds, persistence
counters, anchor snapshots): all of it collapses into `θ̇ = −(M+εI)⁻¹∇L`, `dM = G−λM`.
Boundaries, OOD (a transient adds a decaying rank-1 to `M`), capacity (M's spectrum,
released by `λ`), and gating are one object.  Open: exact-regime / linear-features here;
a deep-network and real-stream instantiation (the flow is natural-gradient/K-FAC-shaped,
so `M` is a running Kronecker-factored second moment) is the next step.
