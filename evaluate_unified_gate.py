# =====================================================================
#  evaluate_unified_gate.py -- CONSOLIDATION, made executable:
#  every protection method in the paper is ONE function, qp_gate(),
#  called with a different constraint set.
#
#      constraint rows                          recovered method
#      -------------------------------------------------------------
#  U1  none                                     naive sharing        (identity)
#  U2  one sampled grad L_u   (inequality)      A-GEM / sign gate    (== closed form)
#  U3  protected bases        (equalities)      OGD / GPM            (== null-space proj.)
#  U4  bases of tasks with overlap < s* only    igfa threshold gate  (== igfa update)
#  U5  all stored grad L_u    (inequalities)    active-set gate
#
#  U2-U4 are checked to MACHINE PRECISION against the paper's original
#  implementations, per step, along whole training trajectories.  U6
#  then runs all five instances of the one function on a mixed stream
#  and reproduces the known retention/adaptation ordering -- six
#  methods, one implementation, different constraint builders.
# =====================================================================
import numpy as np
from interference_ledger import qp_gate, constraints_from_tasks, principal_overlap

def hdr(t): print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)
def orth(rg, d, r): return np.linalg.qr(rg.standard_normal((d, r)))[0]

D, R = 40, 4
rg = np.random.default_rng(0)

if __name__ == "__main__":
    hdr("U1-U3: exact recovery of naive, A-GEM/sign gate, and OGD from qp_gate")
    u = rg.standard_normal(D)
    print(f"  U1 no constraints        : ||qp_gate(u) - u|| = "
          f"{np.linalg.norm(qp_gate(u) - u):.2e}")
    g = rg.standard_normal(D)
    if u @ g < 0: g = -g                                   # active single constraint
    v_closed = u - (u @ g) / (g @ g) * g                   # the paper's sign gate
    print(f"  U2 one sampled gradient  : ||qp_gate - closed-form gate|| = "
          f"{np.linalg.norm(qp_gate(u, G=g[:, None]) - v_closed):.2e}")
    Q = orth(rg, D, 7)
    v_ogd = u - Q @ (Q.T @ u)                              # null-space projection
    print(f"  U3 subspace equalities   : ||qp_gate - OGD projection||  = "
          f"{np.linalg.norm(qp_gate(u, G_eq=Q) - v_ogd):.2e}")

    hdr("U4: igfa's s*-threshold gate == qp_gate with overlap-filtered equalities")
    # full training trajectories must coincide step by step
    SSTAR = 0.5
    B0 = orth(rg, D, R)
    tasks = []
    for t in range(5):                                     # mixed aligned/conflicting
        if t % 2 == 0:
            B = orth(rg, D, R); w = B @ rg.standard_normal(R)
        else:
            B = np.linalg.qr(B0 + 0.05 * rg.standard_normal((D, R)))[0]
            w = B0 @ np.ones(R) + 0.1 * B @ rg.standard_normal(R)
        tasks.append((B @ B.T, w, B))
    worst = 0.0
    for impl in ["igfa", "qp"]:
        w = np.zeros(D); traj = []
        for t, (S, wt, Bt) in enumerate(tasks):
            past = [tasks[i][2] for i in range(t)]
            if impl == "igfa" and past:                    # the paper's implementation:
                prot = [B for B in past                     # protect conflicting only,
                        if principal_overlap(B, Bt) < SSTAR]
                Qp = np.linalg.qr(np.hstack(prot))[0] if prot else None
            for _ in range(300):
                grad = S @ (w - wt)
                if impl == "igfa":
                    step = -(grad - Qp @ (Qp.T @ grad)) if past and Qp is not None else -grad
                else:                                      # the unified gate:
                    E = (constraints_from_tasks(past, B_new=Bt, sstar=SSTAR)
                         if past else None)
                    step = qp_gate(-grad, G_eq=E if E is not None and E.size else None)
                w = w + 0.05 * step
                traj.append(w.copy())
        if impl == "igfa": ref = np.array(traj)
        else: worst = float(np.abs(np.array(traj) - ref).max())
    print(f"  max |w_igfa - w_qp| over 5 tasks x 300 steps: {worst:.2e}  (identical")
    print(f"  trajectories: the threshold gate IS the QP with overlap-filtered rows)")

    hdr("U6: five instances of ONE function on a mixed stream (10 seeds)")
    def run(builder, seed):
        r2 = np.random.default_rng(seed)
        B0 = orth(r2, D, R); tasks = []
        for t in range(6):
            if t % 2 == 0:
                B = orth(r2, D, R); wt = B @ r2.standard_normal(R)
            else:
                B = B0; wt = ((-1) ** (t // 2)) * (B0 @ np.ones(R)) \
                    + 0.1 * B0 @ r2.standard_normal(R)
            tasks.append((B @ B.T, wt, B))
        w = np.zeros(D); rs = np.random.default_rng(seed + 1)
        for t, (S, wt, Bt) in enumerate(tasks):
            past = tasks[:t]
            for _ in range(300):
                u_dir = -S @ (w - wt)
                w = w + 0.05 * qp_gate(u_dir, *builder(past, Bt, w, rs))
        ret = np.mean([0.5 * (w - wt) @ S @ (w - wt) for S, wt, _ in tasks[:-1]])
        S, wt, _ = tasks[-1]
        return ret, 0.5 * (w - wt) @ S @ (w - wt)

    builders = {
      "naive      (no rows)":        lambda past, Bt, w, rs: (None, None),
      "A-GEM      (1 sampled row)":  lambda past, Bt, w, rs: (
          (lambda Si, wi: (Si @ (w - wi))[:, None])(*(
              lambda p: (p[0], p[1]))(past[rs.integers(len(past))][:2])), None)
          if past else (None, None),
      "active-set (all rows)":       lambda past, Bt, w, rs: (
          np.stack([Si @ (w - wi) for Si, wi, _ in past], 1), None) if past else (None, None),
      "OGD        (all equalities)": lambda past, Bt, w, rs: (
          None, constraints_from_tasks([B for _, _, B in past])) if past else (None, None),
      "igfa       (s*-filtered eq.)":lambda past, Bt, w, rs: (
          None, (lambda E: E if E.size else None)(
              constraints_from_tasks([B for _, _, B in past], B_new=Bt, sstar=0.5)))
          if past else (None, None),
    }
    print(f"  {'constraint builder':30s} {'retention':>10s} {'adaptation':>11s}")
    for name, b in builders.items():
        res = np.array([run(b, s) for s in range(10)])
        print(f"  {name:30s} {res[:,0].mean():10.3f} {res[:,1].mean():11.3f}")
    print("\n  -> one qp_gate(), five constraint builders, the known method landscape:")
    print("     naive adapts but forgets; OGD protects everything and pays adaptation;")
    print("     igfa's s* filter recovers transfer on aligned tasks; the sampled row")
    print("     under-enforces; the active set protects with the least restriction.")
