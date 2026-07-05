# Benign forgetting, and what a gate should protect

Validated in [`evaluate_benign_forgetting.py`](evaluate_benign_forgetting.py).

## The observation

The interference identity is **distribution-anchored**, and nothing says the
anchor must be the training set. Evaluated on task A's *deployment*
distribution it splits observed forgetting into two parts with independent
signs:

```
knowledge forgetting     =  ΔL_A^test                (harmful)
memorization forgetting  =  ΔL_A^train − ΔL_A^test   (often benign)
```

**The benign regime is real and exactly predicted** (10 seeds, overparameterized
interpolation with label noise): an interfering update that undoes A's overfit
raises train loss by +1.99 (reads as catastrophic forgetting under standard
metrics) while *lowering* test loss by −3.37 — backward transfer through noise
removal. The sign is given by the same identity with the test anchor,
`ΔL_A^test = ⟨Δ, ∇L_A^test⟩ + ½ΔᵀΣ_A^testΔ`, exact to 8e-16.

Implication for evaluation: a train-anchored forgetting measure cannot
distinguish knowledge loss from memorization loss; part of what CL benchmarks
call catastrophic forgetting is implicit unlearning of noise.

## The law: protect what you validated, not what you fit

Every gating/protection method takes a constraint gradient ∇L_A. Anchoring it
on the **training** data protects the memorized noise together with the
knowledge — it blocks benign repair *and* pays adaptation for the privilege.
Anchoring it on **held-out** data protects the validated function only.

Measured (conflict + repair stream, 10 seeds, initial overfit test-L_A = 7.49):

| gate anchor | final test-L_A (knowledge) | final L_B (adaptation) |
|---|---|---|
| naive (no gate) | 25.90 | 0.00 |
| train-anchored | 22.01 | 7.06 |
| **held-out-anchored (M=200)** | **8.33** | **5.43** |

The held-out anchor dominates the train anchor on **both** axes. And the
anchor's own quality gates the gate: at M=30 (< d, the anchor itself
interpolates its noise) the advantage collapses (22.89/2.98) — protection is
only as good as the *validation* of what it protects, the other face of the
estimation/abstention rule.

## Consequences

- **Design rule** for every functional gate, ledger constraint, and micro-cache
  in this repository: constraint gradients must come from held-out data of
  adequate size. The LLM experiments' held-out micro-cache chunks are mandated
  by this law, not hygiene.
- **Evaluation rule**: report forgetting on deployment-distribution anchors;
  train-anchored forgetting conflates knowledge and memorization.
- **Unlearning connection**: deliberate benign forgetting *is* certified
  unlearning of the memorized component; the same anchor split says what will
  be lost (train fit) and what preserved (validated function).
