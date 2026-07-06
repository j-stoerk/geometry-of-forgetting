# The Interference Ledger — reference

A standard, measurable metric set for adaptive models, and the geometry-driven
decisions each metric supports. This turns *forgetting as interference geometry*
from a continual-learning result into a design calculus: decide **in advance**
when to share, separate, replay, expand, or defer, from measured geometry rather
than post-hoc heuristics.

Reference implementation: [`interference_ledger.py`](interference_ledger.py).
Validation (decision quality vs fixed strategies + oracle):
[`demo_interference_controller.py`](demo_interference_controller.py).

---

## The whole method in one sentence

**One state object, one gate, one loop.** The state is a tracked second moment
(or its sketch) plus held-out micro-cache gradients and per-task bases; every
metric below is a read-out of it. The gate is a single QP — every protection
method is a constraint choice. The loop is *measure → attribute → act*: the
panel measures, the [forgetting decomposition](NOTE_forgetting_decomposition.md)
attributes (floor / capacity / control), and each term's sole remedy acts.

### One gate: all protection methods are constraint sets

`qp_gate(u, G, G_eq)` solves `min ‖v−u‖² s.t. Gᵀv ≤ 0, G_eqᵀv = 0` (NNLS dual;
equalities as mirrored pairs). Machine-checked recoveries
([`evaluate_unified_gate.py`](evaluate_unified_gate.py)):

| constraint rows | recovered method | checked |
|---|---|---|
| none | naive sharing | exact |
| one sampled `∇L_u` (inequality) | A-GEM / threshold-free sign gate | 3e-16 |
| all stored `∇L_u` (inequalities) | **active-set gate** (the QP itself) | — |
| protected bases (equalities) | OGD / GPM projection | 1e-15 |
| bases of tasks with overlap < s\* only | **igfa's s\*-threshold gate** | trajectory-identical, 1e-15 |

Constraint *quality* is governed by two rules of the same state object: anchors
must be **held-out and adequately sized** (a train-anchored gate protects
memorized noise — `NOTE_benign_forgetting.md`), and estimates too uncertain to
act on trigger **defer** (bias-corrected floors + null test —
`evaluate_floor_estimation.py`).

### One loop: measure → attribute → act

| step | what happens | remedy per attributed term |
|---|---|---|
| measure | the panel below, one `LedgerRow` per step | — |
| attribute | `L = floor + capacity + control` (each ≥ 0, measurable) | — |
| act | floor → accept or re-design tasks; capacity → expand/route; control → `qp_gate` + knapsack replay in recoverable excess | validated: each intervention moves only its own term |

## Install & quickstart

```bash
pip install -e .          # from this directory (numpy-only core)
python test_ledger.py     # smoke tests
```

```python
import numpy as np
from interference_ledger import InterferenceLedger, InterferenceController

led = InterferenceLedger(dim=768, rank=8, capacity=64)
ctl = InterferenceController(led, memory_budget=16)

led.register_task(features_task0, target=w0)   # protect what matters
led.tracker.update(feature_batch)              # every batch: track geometry
dec = ctl.decide(grad_new, Phi_new=feature_batch,
                 grad_protected=grad_of_protected_loss)
# dec.action in {share, project, replay, expand, defer, skip}; dec.reason says why
row = led.log(grad_new)                        # one LedgerRow per step: the panel
```

Cross-entropy / LLM streams: `kl_interference(p_old, p_new)` is the exact CE
forgetting on a micro-cache, and `jsd_floor(pi_a, pi_b, dens_a, dens_b)` the
closed-form irreducible floor (see `NOTE_bregman_generalization.md`).

---

## Core metrics — read-outs of the one state object

The state is `(Σ̂, {∇L_u micro-caches}, {B_t})`: the tracked second moment
`Σ_A = E_{D_A}[φφᵀ]` (frozen features) or `E[JJᵀ]` (Gauss–Newton, drifting) or
its Frequent-Directions sketch; held-out per-task gradient caches; per-task
top-`r` bases. Every metric below — and every constraint set of the unified
gate — is computed from these three, so staleness, small-sample bias, and
anchor validity are properties of *one* estimator, not per-method caveats.
Notation: `U` = top-`r` occupied basis, `g` = update/gradient, `λ` = spectrum.

| Metric | Definition | Units | Decision it supports | Cost | Cadence |
|---|---|---|---|---|---|
| **Interference energy** | `½ gᵀΣ_A g` | loss | how much a raw update would harm A | O(dr) | per step |
| **Interference rate** | `⟨g, ∇L_u⟩` (sign) | loss/step | **share vs project** (threshold-free) | O(d) + 1 grad | per step |
| **Shared density** | fraction of stored tasks with `‖UₜᵀU_new‖²_F/r ≥ s*` | — | stream-level: is a gate worth deploying | O(d r T) | per task |
| **Max overlap** | `maxₜ ‖UₜᵀU_new‖²_F/r` ∈ [0,1] | — | fallback share/project when no ∇L_u | O(d r T) | per task |
| **Distortion floor** | `½ Σ_shared (uᵀ(w_new−w_u))²` | loss | replay vs accept the floor | O(d r T) | per task |
| **Occupied rank** | `exp(H(λ/Σλ))` (effective rank) | dims | expand vs continue | O(d²) or O(ℓd) | per task |
| **Free capacity** | `(cap − occupied)/cap` ∈ [0,1] | — | expansion trigger | O(1) | per task |
| **Drift velocity** | `‖P_{Uₜ}P_{Uₜ₋₁} − P_{Uₜ₋₁}P_{Uₜ}‖_F` | — | boundary / shift detection | O(d r²) | per step/task |
| **OOD ratio** | `‖UUᵀg‖/‖g‖` ∈ [0,1] | — | skip noise/OOD batches | O(dr) | per step |
| **Replay value** | recoverable excess `max(damage − floor, 0)`; replay lands AT the floor, never below | loss | where to spend rehearsal: greedy in value/cost is knapsack-optimal | O(d r T) | per task |
| **Confidence** | warmup × drift-penalty ∈ [0,1] | — | conservative defaults when low | O(1) | per step |

**Core vs diagnostic.** Core (required for control): interference rate, drift
velocity, OOD ratio, occupied rank, confidence. Diagnostic (pre-flight / audit):
shared density, distortion floor, max overlap, replay value.

---

## Action set

| Class | Action | State needed | Cost consumed | Composability |
|---|---|---|---|---|
| Update | **share** | ∇L_u or overlap | — (uses all transfer) | exclusive with project |
| Update | **project / separate** | occupied `U` | (recoverable) adaptation | exclusive with share |
| Update | **defer** | occupied `U`, confidence | adaptation, revisit later | fallback for share |
| Memory | **replay** | stored `Σ_A`/exemplars/dream state | memory budget | composes with share/project |
| Arch | **expand** | occupied rank, capacity | rank/params/compute | composes with any update |
| Arch | **route / merge** | per-task heads | latency | sequential |
| Control | **skip** | OOD ratio, grad scale | — | precedes all |

---

## Decision rules (controller)

Order per step — **detect shift → reject OOD → estimate geometry → pick action → update state**:

1. **skip** if `ood_ratio < 0.5` **and** `‖g‖ > 3·(running grad scale)` — orthogonal-and-extreme ⇒ noise/OOD.
2. **expand** if `free_capacity < ε` — occupied rank near capacity.
3. **defer (protect)** if `confidence < 0.5` — a wrong *share* costs retention; a wrong *project* costs only recoverable transfer, so protect when unsure.
4. **project** if `interference_rate < 0` (measured conflict); offer **replay** first if `floor > 0` and memory is available.
5. **share** otherwise (measured transfer, `interference_rate ≥ 0`).

The default share/project decision is the **threshold-free functional gate**
(sign of measured interference); the similarity-`s*` gate is the fallback when no
protected gradient is available.

---

## Regime matrix — where each metric is exact / bounded / empirical / heuristic

| Metric | Frozen-feature (A1) | Function space (realizable) | Param-space, drift | Large model / LLM |
|---|---|---|---|---|
| Interference energy | **exact** (Thm 1) | **exact** as `½‖Δf‖²_{L²}` | first-order (path-avg `H̄`) | empirical proxy |
| Interference rate | **exact** | **exact** (path integral, Sec. S15) | exact continuous / O(η) discrete | empirical (micro-cache ∇L_u) |
| Overlap / density | **exact** | exact (function support) | drifts; re-estimate | sketch-estimated |
| Distortion floor | **exact** (Thm 4) | **exact** (density form, Cor N3) | bounded below | heuristic estimate |
| Occupied rank | exact | exact | recursive estimate | FD-sketch estimate |
| Drift velocity | n/a (static) | exact | **empirical** signal | empirical |
| OOD ratio | exact | exact | empirical | empirical |

**Measurable failure indicators.** Stale-basis error: rising `drift_velocity`
with flat estimate ⇒ re-estimate. Margin violation (ReLU region-exactness):
preactivation margin `< 0` on the A-support ⇒ region certificate void. Estimator
noise: `occupied_rank` jitter across steps. Small-data probe: wide floor CI ⇒
lower confidence, prefer defer.

---

## Estimation & update

- **Geometry tracker** (`GeometryTracker`): `recursive` decayed covariance
  (`C ← γC + mean(φφᵀ)`, O(d²)) or `fd` Frequent-Directions sketch (O(ℓd),
  `ℓ ≥ 2r` — the validated sizing rule). Decay `γ ≈ 0.6`; update per batch.
- **Interference rate** at scale: `∇L_u` from a micro-cache (a few thousand
  tokens/domain), the anchor construction, or the dream-state second moment.
- **Floor estimation at small n**: plug-in floors are biased *upward* ~1/n
  (noise reads as disagreement — identical tasks look conflicting at
  micro-cache sizes). Use the Miller–Madow entropy correction and flag
  conflict only above the 95th percentile of the exchangeability null
  (resample all tasks from the pooled conditional); otherwise **defer**.
  Validated: false-conflict 100/100 → 12/100 at n=150, power 95/100
  (`evaluate_floor_estimation.py`).
- **Confidence** calibration: `warmup_steps ≈ 20` for per-batch training,
  `≈ 3–5` for per-task streams. Known limitation: a clean task boundary raises
  `drift_velocity` (correct as a *boundary* signal) which currently also lowers
  confidence; separating "boundary detected" from "estimate unreliable" is a
  planned refinement.

---

## Status

Validated: on a labeled synthetic stream with a regime oracle, the controller
attains lower mean retained loss than always-share, always-project, and
always-replay, with the costly error (false-share) near zero and decisions erring
toward protection under uncertainty. Extends to real streams via the torch
adapters (cached ViT features, LoRA gradients) used elsewhere in this repository.
