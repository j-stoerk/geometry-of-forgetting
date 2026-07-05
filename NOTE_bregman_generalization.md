# Beyond squared loss: the Bregman generalization of the interference calculus

The paper's exact function-space results are stated for squared loss, while its
LLM and classification experiments run on softmax cross-entropy. This note
removes that mismatch: **every core identity holds exactly for the whole
canonical (Bregman / exponential-family) loss family**, with closed forms for
the CE case. Validated to machine precision in
[`evaluate_bregman_identity.py`](evaluate_bregman_identity.py).

Setup: canonical loss `ℓ(y,f) = φ(f) − yᵀf`, φ convex (squared: `φ=½‖f‖²`,
`μ=f`; softmax-CE: `φ=logsumexp`, `μ=softmax`; logistic, Poisson, … likewise),
`L_A(f) = E_A[ℓ(y, f(x))]`, `μ_A = μ(f_A)`, predictive distributions
`p_A = softmax(f_A)` etc.

## T1'' — Exact Bregman interference identity (any architecture, any finite change)

By the definition of the Bregman divergence `D_φ(b,a) = φ(b) − φ(a) − μ(a)ᵀ(b−a)`:

```
ΔL_A  =  E_A[(μ_A − y)ᵀ Δf]  +  E_A[ D_φ(f_B, f_A) ]          (exact, no Taylor)
```

- Squared loss: `μ_A−y = r_A`, `D_φ = ½‖Δf‖²` → the paper's identity verbatim.
- Softmax-CE: `D_φ(f_B, f_A) = KL(p_A ‖ p_B)` **exactly** (validated 3e-16).

**Converged anchor** (`μ_A = y` on supp D_A): `ΔL_A = E_A[KL(p_A‖p_B)] ≥ 0` —
*forgetting is the expected KL divergence between the old and new predictive
distributions on the protected data* (validated 2e-16). The interference
energy's general form is a Bregman divergence; ½‖Δf‖² is its Gaussian case.

**Dichotomy**: learning B is lossless iff `p_B = p_A` a.e. on supp D_A — note
the kernel is *larger* than for squared loss: per-sample constant logit shifts
are invisible to softmax (validated exactly 0).

## The floor is a Jensen–Shannon divergence (closed form)

For one function serving two tasks with input densities `p_A, p_B` and
conditionals `π_A(·|x), π_B(·|x)`, the pointwise irreducible excess risk

```
min_q  p_A·KL(π_A‖q) + p_B·KL(π_B‖q)
```

is attained at the **density-weighted mixture** `q* = (p_Aπ_A + p_Bπ_B)/(p_A+p_B)`
with value `(p_A+p_B) · JSD_w(π_A, π_B)`, `w = p_A/(p_A+p_B)` (validated 7e-16
against numeric minimization). Hence

```
floor  =  ∫ (p_A(x)+p_B(x)) · JSD_w(x)( π_A(·|x), π_B(·|x) ) dx
```

The paper's quadratic floor `½∫ p_Ap_B/(p_A+p_B)(y_A−y_B)² dx` is the
small-divergence / Gaussian limit (`JSD_w ≈ w(1−w)·χ²/2`). The optimal joint
conditional is the mixture — the information-geometric analogue of the
Σ-weighted mean in Theorem "optimal merge".

## What transfers and what is squared-loss-specific

| Result | Bregman/CE status |
|---|---|
| Interference identity | **exact** (T1'') |
| Interference energy formula | Bregman divergence (KL for CE); quadratic form is the Gaussian case |
| Removability dichotomy | **exact**; kernel enlarged (softmax invariances) |
| A1 finite-step dichotomy | **exact for any loss** (kernel step ⇒ Δf=0 ⇒ ΔL=0 at any η; validated at ‖ΔW‖=10) |
| A1 finite-step *energy* | squared-loss-specific (CE loss is not quadratic in w; still computable in closed form since Δf = ΔᵀΦ exactly) |
| Distortion floor | **closed form**: density-weighted JSD |
| Optimal merge | mixture of conditionals replaces Σ-weighted mean |
| Monotone gate + QP corollary | **unchanged** — they act on `⟨g, ∇L_A⟩ = dL_A/dt`, never used squared loss |
| Leakage law | quadratic term: residual-weighted; converged-anchor order inherits from Fisher-weighted curvature (not re-derived here) |

## The T-task floor is a conditional mutual information (exact)

Generalizing the JSD floor to T tasks (uniform prior) and rewriting the
generalized Jensen–Shannon divergence as an information quantity
([`evaluate_info_floor.py`](evaluate_info_floor.py), all checks ≤ 6e-16):

```
D*_T  =  Σ_x Σ_t p_t(x) KL(π_t ‖ π̄_x)  =  T · I(T ; Y | X)
      =  T · [ H(Y|X) − H(Y|X,T) ]
```

with the minimizer the density-weighted mixture `π̄`. The floor of a single
predictor over a task family **is** the conditional mutual information between
task identity and label given input — the label-entropy cost of hiding the
task. Removability ⟺ `I(T;Y|X) = 0` ⟺ conditionals agree wherever supports
overlap. The paper's Theorem (mi) proves a *sandwich* of entropy bounds for the
quadratic case; under cross-entropy the sandwich closes to an identity. A
trained shared softmax head converges to this floor from above (gap 6e-16), so
it is attained, not merely bounded. Operationally, for an LLM domain stream:
*the irreducible single-model floor equals I(domain ; next-token | context)*,
estimable from micro-caches as mean per-domain vs mixture log-likelihood ratio.

## The retroactive payoff

The GPT-2 / pythia functional-gate experiments gate on cross-entropy gradients.
Under the squared-loss-only theory they were "the mechanism transferred
empirically"; under T1'' they are **the exact monotone controller for the loss
actually being trained** — the sign test reads `dL_A/dt` exactly, and the
active-set/NNLS gate is the exact KKT solution of the CE monotone-retention QP.
The measured pythia-410m active-set result (forgetting −36% vs naive at best
final loss) is therefore theory-matched end to end: identity (exact), gate
(optimal), enforcement (full active set), outcome (measured).

Also immediately available: the pre-flight panel for LLM streams gains an exact
target — shared-support conditional disagreement `E[KL]`/JSD between domains,
estimable from micro-caches, replacing the input-overlap proxy that fails on
text.
