# The forgetting decomposition: incompatibility + capacity + control

A bias/variance-style **attribution theorem for continual learning**, validated
in [`evaluate_forgetting_decomposition.py`](evaluate_forgetting_decomposition.py).

For a stream served by one model, define three reference levels:

| Level | Meaning | How measured |
|---|---|---|
| `D*_∞` | best achievable by *any* single predictor — the **floor** | from data alone: `T·I(T;Y\|X)` (CE) / quadratic form; zero for input-disjoint tasks |
| `D*_class` | best achievable in the **deployed class** (width/rank/architecture) | class optimum at deployed capacity (closed form in the exact regime; estimable from the spectrum) |
| `L` | what the deployed **training policy** actually achieved | observed |

Then, with every term ≥ 0 and the sum exact:

```
L  =  D*_∞                (INCOMPATIBILITY -- property of the task family;
                           no gate, replay, or width removes it)
    + [D*_class − D*_∞]   (CAPACITY -- property of the model class;
                           only widening/expansion/routing removes it)
    + [L − D*_class]      (CONTROL -- property of the training policy;
                           only a better gate/replay/ordering removes it)
```

The identity is elementary; the content is that (a) each term is **measurable**
(floor from data, class optimum from the deployed geometry, control as the
remainder) and (b) **interventions move only their own term** — which makes the
decomposition a diagnosis:

## Validation (exact regime, 10 seeds per stream)

| stream | floor | capacity | control |
|---|---|---|---|
| conflict, ample width, full replay | **12.39** | 0.00 | 0.00 |
| disjoint tasks through width-12 features | 0.00 | **5.35** | 0.00 |
| mixed (conflict + naive training) | **5.42** | 0.00 | 4.04 |

Intervention attribution (6 seeds):
- **Policy upgrade** (naive → active-set gate), mixed stream: floor 6.23 → 6.23,
  capacity 0 → 0, control 4.65 → 2.15. Only control moves.
- **Widening** (12 → 24 features), capacity stream: capacity 6.02 → 0.00
  (100% removed), control ≈ unchanged. Only capacity moves.
- **Any policy** (naive / active-set / replay) on the width-12 stream: capacity
  6.02 untouched by all three; control 12.37 / 11.77 / 0.00. Policies cannot
  buy width.

Two conceptual points the construction forces into the open:

1. **Capacity is representation width, not subspace rank of the final vector.**
   "Best rank-R subspace containing the solution" collapses (any single optimum
   spans one dimension). Capacity binds when *compatible* tasks (function-space
   floor exactly 0) are squeezed through a too-narrow feature map and collide
   in the deployed class — the paper's capacity knee, now sited precisely as
   the middle term.
2. **Perturbed subspaces are not conflict.** In the quadratic regime, tasks on
   slightly-perturbed copies of a subspace are *jointly satisfiable* (24 soft
   constraints in 48 dims → floor exactly 0); genuine conflict requires exactly
   shared support. Real conflict in practice is the CE case, where the floor is
   the conditional mutual information — nonzero whenever conditionals disagree
   on overlapping support, no knife-edge.

## The prescription

Measure which term dominates **before choosing a method family**: a
floor-dominated stream needs task re-design or acceptance (no method helps); a
capacity-dominated stream needs width/routing (no gate or replay helps); a
control-dominated stream needs a better policy (the QP gate, replay where the
[recoverable excess](evaluate_replay_policy.py) is high). The method taxonomy
(`demo_method_taxonomy.py`), the knapsack replay policy, and the pre-flight
protocol (`PREFLIGHT.md`) are the three terms' respective toolkits.
