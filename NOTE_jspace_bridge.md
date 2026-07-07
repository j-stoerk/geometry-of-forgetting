# J-space / interference bridge: a first probe (mixed, honest)

`jspace_interference_results.json` (kaggle_jspace_interference_exact.py, user rework
adding H1-grad and H1-KL; frozen pythia-410m, readout layer 16/24, rank-64 subspaces).

**Correlational: modest support.**
- J-space vs Sigma_A-subspace principal overlap: mean(cos^2)=0.120 vs 0.0625 random
  baseline (~2x chance); leading principal cosines 0.69, 0.67, 0.63, ... -- the two
  averaged-Jacobian subspaces share real structure but are NOT the same subspace.
- H1 (interference concentration in J-space): gradient-energy fraction 9.3% vs 6.2%
  baseline (1.5x); **output-space KL fraction 14.6% vs 6.2% (2.35x)**.  Forgetting is
  enriched in the verbalizable subspace beyond chance, more so in function/output space
  than in raw gradient geometry -- directionally consistent with the bridge.  But ~79%
  of the interference is orthogonal to the rank-64 J-space.

**Causal: inconclusive (under-powered).**
- H2 forgetting on A: naive 0.126, protect-Jspace 0.128, protect-Sigma 0.119,
  protect-random 0.130.  No rank-64 projection meaningfully protects -- **including this
  paper's own Sigma-subspace (only 5.7% reduction)**.  When the positive control fails,
  the intervention cannot discriminate J-space.  Cause: at this mid-layer the
  interference is high-effective-rank (79% outside any 64-dim subspace), so low-rank
  protection is inherently weak -- consistent with the capacity-knee story, not a clean
  refutation.

**Verdict.** The bridge is correlationally suggestive (2x subspace overlap, 2.35x
output-space interference enrichment) but not causally established here.  A discriminating
causal test needs a regime where low-rank protection is effective (a layer/rank where the
interference effective rank is low, or a larger protected subspace); that is future work.
The paper keeps the bridge as a PROPOSED experiment with this preliminary finding noted.
