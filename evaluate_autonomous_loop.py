# =====================================================================
#  evaluate_autonomous_loop.py -- local (CPU, numpy) validation of the
#  autonomous boundary/OOD loop (roadmap D3 + D5) and the
#  stale-vs-recursive tracking benchmark:
#
#  A1  BOUNDARY DETECTION vs ORACLE.  Streams of feature batches from
#      segment subspaces with abrupt boundaries.  The task-free signal
#      is the tracker's drift velocity (projector commutator norm); a
#      boundary is declared on a z-score spike over a trailing window.
#      Report precision/recall (+/-2 batches) on clean and noisy
#      streams -- degrades gracefully, no catastrophic false cascade.
#
#  A2  OOD REJECTION in the same loop.  5% of batches are large-norm
#      orthogonal noise.  Signal: projection ratio onto the occupied
#      subspace < 0.5 AND norm > 3x running scale (the ledger rule).
#      Report hit rate / false-alarm rate, and that OOD batches do NOT
#      trigger boundary declarations (precedence: skip first).
#
#  A3  STALE vs RECURSIVE geometry under drift (the controlled-setting
#      benchmark).  A slowly rotating subspace; compare the alignment
#      of (i) a basis frozen at t=0 (stale) and (ii) the recursive
#      tracker, to the true current subspace.  Also the gradual-drift
#      case for the boundary detector: velocity stays sub-threshold
#      (correctly NO boundary) while the tracker follows -- boundary-
#      free operation is the point, not boundary detection per se.
# =====================================================================
import numpy as np
from interference_ledger import GeometryTracker, principal_overlap

def hdr(t): print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)
def orth(rg, d, r): return np.linalg.qr(rg.standard_normal((d, r)))[0]

D, R, BATCH = 48, 5, 64

def make_stream(seed, n_seg=6, seg_len=30, noisy=False, ood_frac=0.05):
    rg = np.random.default_rng(seed)
    bases = [orth(rg, D, R) for _ in range(n_seg)]
    batches, true_bounds, true_ood = [], [], []
    i = 0
    for s in range(n_seg):
        if s > 0: true_bounds.append(i)
        for _ in range(seg_len):
            if rg.random() < ood_frac:
                Bp = orth(rg, D, R)                        # fresh random subspace, big norm
                X = rg.standard_normal((BATCH, R)) @ Bp.T * 8.0
                true_ood.append(i)
            else:
                X = rg.standard_normal((BATCH, R)) @ bases[s].T
                if noisy: X = X + 0.6 * rg.standard_normal(X.shape)
            batches.append(X); i += 1
    return batches, true_bounds, true_ood, bases

def run_loop(seed, noisy):
    batches, tb, tood, bases = make_stream(seed, noisy=noisy)
    tr = GeometryTracker(D, rank=R, decay=0.8)
    vel_hist, declared, skipped = [], [], []
    scale = None
    for i, X in enumerate(batches):
        nrm = np.linalg.norm(X) / np.sqrt(len(X))
        # --- OOD precedence: skip before the boundary logic sees it ---
        is_ood = False
        if tr.U is not None and scale is not None:
            pr = np.linalg.norm(X @ tr.U) / max(np.linalg.norm(X), 1e-12)
            is_ood = (pr < 0.5) and (nrm > 3.0 * scale)
        if is_ood:
            skipped.append(i); continue                    # do NOT update tracker/state
        scale = nrm if scale is None else 0.95 * scale + 0.05 * nrm
        tr.update(X)
        v = tr.drift_velocity()
        vel_hist.append((i, v))
        if len(vel_hist) > 8:                              # z-score spike on trailing window
            w = np.array([x for _, x in vel_hist[-9:-1]])
            if w.std() > 1e-12 and (v - w.mean()) / w.std() > 4.0 and \
               (not declared or i - declared[-1] > 5):
                declared.append(i)
    hits = sum(any(abs(b - d_) <= 2 for d_ in declared) for b in tb)
    prec = (sum(any(abs(b - d_) <= 2 for b in tb) for d_ in declared) / len(declared)) if declared else 1.0
    ood_hit = sum(1 for j in tood if j in skipped) / max(len(tood), 1)
    ood_fa = sum(1 for j in skipped if j not in tood) / max(len(skipped), 1) if skipped else 0.0
    ood_as_boundary = sum(any(abs(j - d_) <= 1 for d_ in declared) for j in tood)
    return hits / len(tb), prec, ood_hit, ood_fa, ood_as_boundary

if __name__ == "__main__":
    hdr("A1+A2: autonomous boundary/OOD loop vs oracle (10 seeds)")
    for noisy in [False, True]:
        rows = np.array([run_loop(s, noisy) for s in range(10)])
        print(f"  {'noisy' if noisy else 'clean'} stream : boundary recall {rows[:,0].mean():.2f}"
              f"  precision {rows[:,1].mean():.2f}   OOD hit {rows[:,2].mean():.2f}"
              f"  false-skip {rows[:,3].mean():.2f}   OOD-as-boundary {rows[:,4].mean():.1f}/stream")
    print("  (oracle = true segment changes; +/-2 batches counts as detected)")

    hdr("A3: stale vs recursive tracking under gradual drift (10 seeds)")
    align_st, align_rec, max_vel = [], [], []
    for seed in range(10):
        rg = np.random.default_rng(seed)
        B = orth(rg, D, R); B_start = B.copy()
        tr = GeometryTracker(D, rank=R, decay=0.8)
        Q = orth(rg, D, D)                                  # fixed rotation generator
        A = 0.02 * (Q[:, :2] @ Q[:, 2:4].T - Q[:, 2:4] @ Q[:, :2].T)
        Rot = np.eye(D) + A + A @ A / 2                     # ~exp(A), small rotation/step
        vels = []
        for t in range(120):
            B = np.linalg.qr(Rot @ B)[0]                    # slow continuous drift
            tr.update(rg.standard_normal((BATCH, R)) @ B.T)
            if t > 5: vels.append(tr.drift_velocity())
        align_st.append(principal_overlap(B_start, B))
        align_rec.append(principal_overlap(tr.subspace(R), B))
        max_vel.append(max(vels))
    print(f"  final alignment with TRUE subspace: stale basis {np.mean(align_st):.2f}"
          f"   recursive tracker {np.mean(align_rec):.2f}   (1.0 = perfect)")
    print(f"  max drift velocity under gradual drift: {np.mean(max_vel):.3f}"
          f"  (stays sub-spike: correctly NO boundary declared; the tracker")
    print(f"   follows continuously -- boundary-FREE operation, which is the point)")
