# J-space / interference bridge: RESOLVED (enrichment, not identity)

Two runs settle the cross-paper bridge (this paper's interference geometry vs the
Jacobian-lens J-space of Gurnee, Lindsey et al. 2026), on frozen pythia-410m.

## The strong claim (J-space IS the interference subspace) is REFUTED
`kaggle_jspace_layers.py` -> `jspace_layers_results.json`, 11 layers, geometry only:
the interference Sigma_A = E[g g^T] (g = d CE_A / d h_l) is HIGH-effective-rank at
EVERY layer -- participation ratio 150-300, and 80% of its energy needs >= 285 of 1024
dimensions (min 285, layer 2).  A workspace-sized subspace (~25-64 dims) therefore
cannot contain the bulk of forgetting.  This also explains why the earlier causal H2 was
inconclusive: no low-rank projection can protect (not even Sigma's own top-64, 6%),
because the interference genuinely lives in hundreds of dimensions -- not bad luck, a
geometric fact.

## The weak claim (J-space is ENRICHED for interference) is CONFIRMED, and grows with depth
- J-space carries 1.33x (layer 2) -> 2.49x (layer 22) its dimensional share of
  interference energy; enriched at 11/11 layers; corr(depth, enrichment)=0.86.
- J-space overlap with Sigma_A's own top-32 subspace: 0.050 -> 0.152 (baseline 0.031),
  also rising with depth (1.6x -> 4.9x chance).
- Consistent with the earlier direct probe (H1-KL at layer 16: output-space interference
  2.4x enriched in J-space).

## Reading
The two averaged-Jacobian objects share real, increasing-with-depth structure -- the
verbalizable subspace is over-represented in the forgetting geometry, more so toward the
readout -- but they are NOT the same subspace: forgetting is high-dimensional, the
workspace is low-dimensional.  Honest, robust, and directional: a component relationship,
not an identity.  Near the output, representations become both more verbalizable and more
interference-bearing, which is a coherent (and independently testable) mechanistic story.
