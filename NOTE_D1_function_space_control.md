# D1 — Boundary-free function-space control: the unified theorem package

The roadmap's item-2 deliverable: one consolidated statement of what is exact,
what leaks, and what the controller should do, in function space, without task
boundaries. Validated locally by
[`evaluate_functional_control.py`](evaluate_functional_control.py) (leakage law,
QP equivalence, constraint sampling) and
[`evaluate_autonomous_loop.py`](evaluate_autonomous_loop.py) (autonomous
boundary/OOD loop, stale-vs-recursive tracking).

Notation: model `f(·;θ)`, task losses `L_i(θ) = ½ E_i[(f(x;θ) − y_i(x))²]`,
residual `r_i = f − y_i`, per-sample tangent `∇f(x)`, Gauss–Newton metric
`G_i(θ) = E_i[∇f ∇fᵀ]`, curvature along `v`: `κ_v(x) = vᵀ∇²f(x) v`.

---

## T1 — The primary object (identity; exact, any architecture)

For any training procedure taking `f_A → f_B`:
`ΔL_A = E_A[Δf·r_A] + ½E_A[Δf²]`, with the frozen-feature (A1) results as the
linear special case `Δf = Δᵀφ`. (Paper Thm. "function-space interference
identity"; not restated here.)

## T2 — Retention: exact in continuous time, quantified leakage in finite steps

**Conservation (exact).** If the training velocity satisfies `G_A(θ_t) θ̇ = 0`
then `∇f(x)ᵀθ̇ = 0` for `D_A`-a.e. `x`, hence `dL_A/dt = E_A[r_A·∇fᵀθ̇] = 0` —
regardless of the residual, the architecture, or drift.

**Leakage law (new).** For a finite step `θ⁺ = θ + ηv` with `v ∈ ker G_A(θ)`,
expanding `L_A` along the ray (all identities via `∇fᵀv ≡ 0` on supp `D_A`):

```
ΔL_A(η) = (η²/2)·E_A[r_A κ_v]  +  (η⁴/8)·E_A[κ_v²]  +  h.o.t.
```

| Regime | Leading leakage | Validated |
|---|---|---|
| Frozen-linear features (A1) | **0 at all orders** — finite-step exact | max \|ΔL\| = 1e-29 up to η=10 |
| Converged anchor (`r_A = 0`) | **quartic**: `(η⁴/8)E[κ_v²] ≥ 0` | slope 4.03, coeff ratio 1.008 |
| Non-converged anchor | quadratic: `(η²/2)E[r_A κ_v]` | slope 2.00, coeff ratio 0.998 |

The cubic term also vanishes at a converged anchor (both its parts carry `∇fᵀv`
or `r_A`). Reading: "A1 buys finite-step exactness" is now quantitative — the
step-size penalty for leaving A1 is *fourth order* if you protect from converged
anchors, and the predictor–corrector retraction (paper Sec. B5) removes even
that (anchor forgetting ~1e-14 independent of η). This also yields a
retention-aware step-size rule: to keep leakage below ε, take
`η ≤ (8ε / E_A[κ_v²])^{1/4}` at converged anchors.

## T3 — Control: the gate is the *optimal* monotone controller (new)

Define the control objective (function-space monotone retention): follow the
desired update `u = −∇L_new` as closely as possible without harming protected
tasks:

```
v* = argmin_v ½‖v − u‖²   s.t.   ⟨∇L_i, v⟩ ≤ 0   for protected i = 1..m.
```

KKT gives `v* = u − Gλ` with `G = [∇L_1 … ∇L_m]` and `λ = NNLS(G, u)` — the
dual is a nonnegative least squares. Consequences (each machine-checked):

- **m = 1**: `v*` is *exactly* the threshold-free functional sign gate
  (‖v_QP − v_gate‖ = 0). The gate is not a heuristic; it is the closed-form
  solution of the single-constraint control problem.
- **No violated constraints**: `λ = 0`, `v* = u` — naive sharing recovered.
- **All constraints violated**: `v*` is feasible (max rate 4e-15 ≤ 0) but
  **dominates equality projection (OGD/GPM)**: projection pins protected rates
  at exactly 0, while the QP additionally permits protected-loss *decrease*
  (measured min rate −1.27) and stays closer to the desired update
  (‖v*−u‖ = 2.742 ≤ ‖v_proj−u‖ = 2.753). Projection is the QP with
  inequalities hardened to equalities — strictly more restrictive, never
  better in the exact regime.
- The gate therefore **interpolates** naive ↔ projection through the active
  set, which answers "when does it recover each" exactly: sharing when no
  constraint is active, projection-like protection when all are.

## T4 — Constraint sampling explains the measured vision gap (new)

Sampling **one** protected task per step (the scalable `func-sign`
implementation) enforces one row of the QP per step: stochastic
under-enforcement. Exact-regime measurement (dense-conflict stream, 10 seeds):
single-probe gating recovers **35%** of projection's retention gain; enforcing
the full measured active set recovers **100%** — while on aligned streams both
gates keep naive's adaptation (constraints inactive). This is the mechanism
behind the C100-superclass result (sign gate recovered ~18% there, with noisier
micro-cache constraints and Adam): *the gate did not fail; the sampling did.*

**Prescription — active-set gating**: per step, compute all m interference
rates (m dot products against cached `∇L_i`; m is the number of *stored*
protected tasks, typically ≤ tens), then one NNLS on the violated subset
(m-column system, negligible cost). Cost scales with m gradients cached, not
with parameters. Predicted to close most of the superclass gap at scale;
untested there — a GPU run would be needed to confirm.

## Autonomous loop (D3/D5 validation, local)

One loop, fixed precedence: **skip OOD → update tracker → detect boundary →
gate/protect**. Signals: OOD = projection ratio < 0.5 onto the occupied
subspace AND norm > 3× running scale (skipped batches never update state);
boundary = z-score spike (>4σ over a trailing window) of the tracker's drift
velocity. Measured vs oracle (10 seeds): boundary recall/precision
**1.00/0.98** clean, **0.96/0.94** noisy; OOD hit rate 0.98–0.99 with **zero**
false-skips. Under *gradual* drift the velocity correctly stays sub-spike (no
boundary exists to declare) while the recursive tracker follows continuously —
final alignment with the true subspace **1.00** vs **0.77** for a stale basis.
Boundary-free operation is not merely detection: where no boundary exists, the
tracker's continuous following *is* the correct behavior, and protection quality
is set by tracker alignment (the stale-vs-recursive benchmark of the roadmap).

## Regime boundaries (what holds where)

| Result | Continuous time | Finite step, converged anchor | Finite step, non-converged | Estimated constraints |
|---|---|---|---|---|
| Conservation / monotone retention | exact, any arch. | O(η⁴), coeff `E[κ²]/8` | O(η²), coeff `E[rκ]/2` | enforce margin `⟨∇L_i,v⟩ ≤ −δ` with δ ~ estimator noise |
| Gate optimality (T3) | exact (KKT) | inherits leakage above | inherits | NNLS on noisy G: conservative under the ledger's confidence rule |
| A1 (frozen-linear) | exact | **exact at all η** | exact | exact |
