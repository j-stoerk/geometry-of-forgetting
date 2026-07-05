# pythia-410m, v3.2/v3.3 arm set with func-active — 2026-07-05

Provenance: per-seed JSON lost (session ended before download); values below
reconstructed from the console log (means ± np.std over 3 seeds, LR 1e-4,
stream news→imdb→wikitext→tweets, LoRA r=8, 250 steps/domain). The naive arm
reproduces the archived valid run (3.624/0.321) to within one std.

| arm            | final loss      | forgetting      |
|----------------|-----------------|-----------------|
| naive          | 3.625 ± 0.002   | 0.324 ± 0.002   |
| protect-all    | 3.617 ± 0.006   | 0.319 ± 0.008   |
| func-sign-adam | 3.605 ± 0.004   | 0.300 ± 0.004   |
| **func-active**| **3.536 ± 0.005** | **0.206 ± 0.005** |

Welch t-tests from summary stats (unpaired — conservative, since arms share
seeds): func-active vs every other arm, both metrics, p ≤ 4.6e-4; all gaps
exceed 14 standard errors. Forgetting −36% vs naive, −31% vs the sampled
gate; best final loss of all arms (better adaptation AND retention).
Runtime overhead vs the sampled gate: +21% (2228s vs 1842s per 3 seeds).

Reading: this is the QP corollary's prescription confirmed at scale. The
sampled sign gate enforces one row of the monotone-retention QP per step
(stochastic under-enforcement — the mechanism measured in the exact regime at
35% recovery and in vision at ~18%); enforcing the full active set (all past
domains' micro-cache gradients + NNLS) recovers the rest and dominates naive,
subspace protection, and the sampled gate on both axes simultaneously.

Still pending: the vision func-active run (kaggle_sign_gate_vit.py v2.1) —
predicted to close the C100-superclass gap; and per-seed JSON for exact paired
tests (optional re-run with v3.3-resume).
