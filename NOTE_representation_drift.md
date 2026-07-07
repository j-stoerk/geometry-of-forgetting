# The most pressing gap toward true CL: stale geometry under representation drift

Everything exact in this framework assumes A1 (frozen features): it protects a stored
second moment Sig_A = E[phi_A phi_A^T].  True continual learning updates the backbone,
so the features phi(.;theta) DRIFT and a stored Sig_A describes a feature space that no
longer exists.  This is the deepest gap between the paper's exact regime and full-network
CL.  Two geometric identities fix it -- validated on a drifting 2-layer net
f(x)=v^T tanh(Wx)  (`evaluate_representation_drift.py`).

## I1 -- function-space invariance is drift-proof; the stored parametric form is stale
Delta L_A = 1/2 E_A[(f_now - f_A)^2], recomputed on A's stored inputs in the CURRENT
model, tracks true forgetting to ~4 decimals through progressive backbone drift, while
the stored parametric form 1/2 Delta^T G_A^stale Delta is systematically ~45% wrong
because it uses the vanished feature space.  Forgetting is a property of the FUNCTION,
not of a fixed parameter geometry.

## I2 -- the covariant-transport identity
If features drift phi -> T phi, the geometry pushes forward as Sig_A -> T Sig_A T^T.
Measuring T by a cheap regression (phi_now ~ T phi_old on the anchors), the transported
stored geometry matches the current geometry to 0.5% (||S_now - T S_old T^T||/||S_now||),
and its top-subspace overlap with the current geometry is 0.98 versus 0.85 for the
un-transported (stale) one.  Transport realigns a stored metric with drifted features.

## Control -- online protection under drift
A-forgetting after learning B on the drifting net:
  naive                          1.37
  stale-null (stored subspace)   0.45   (decays: the subspace is stale)
  functional-null (recomputed)   0.09   (drift-proof: 5x better than stale, 16x vs naive)
Protecting the STORED subspace decays under drift; protecting the FUNCTION on stored
inputs -- recomputing the anchor Jacobians in the current model -- stays protective.

## The rule, and how it ties the paper together
Anchor FUNCTIONALLY (re-evaluate the constraint on stored inputs) or TRANSPORT the
metric by the measured feature map T; do NOT protect a frozen stored geometry once the
backbone moves.  This is the mechanism behind three existing results: the LLM micro-cache
gate works at scale precisely because it RECOMPUTES grad L_A each step (drift-proof),
whereas stored-subspace methods drift; the held-out-anchor / benign-forgetting law is the
same "anchor functionally" prescription; and the memory-metric flow gains a drift term --
transport M by T (dM = G_t - lam M + [T,M]-type correction) -- to stay aligned as the
representation moves.  Open: a deep-network instantiation of the transport at scale.
